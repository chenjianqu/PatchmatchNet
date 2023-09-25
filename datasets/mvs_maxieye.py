import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
import os
import random

from PIL import Image

from datasets.data_io import read_cam_file, read_image, read_pair_file, scale_to_max_dim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

from models.module import generate_pointcloud
from tools.colmap_utils import read_model_get_points
from tools.pose_utils import convert_rvector_to_pose_mat, convert_rollyawpitch_to_rot, convert_quat_to_pose_mat
from tools.visualize_utils import vis_points, write_ply
from utils import to_cuda


def get_data_param(path_idxs, is_train=True):
    data_infos_list = []
    param_infos_list = []
    image_paths_list = []
    idxs = []
    for sce_id, path in enumerate(path_idxs):
        # 读取内参和外参
        param_path = path + '/gen/param_infos.json'
        with open(param_path, 'r') as ff:
            param_infos = json.load(ff)
            param_infos_list.append(param_infos)

        # 图像路径
        image_paths = os.path.join(path + '/org/img_ori', "*.jpg")
        image_paths_list.append(image_paths)

        # 加载每个图像的位姿等信息
        with open(path + '/gen/bev_infos_roadmap_fork.json', 'r') as ff:
            data_infos = json.load(ff)
            data_infos_list.append(data_infos)

        idxs += [[sce_id, index] for index in range(len(data_infos))]
    return param_infos_list, data_infos_list, image_paths_list, idxs


def project_to_pixel(T_cw: np.ndarray, K: np.ndarray, xyz_Pw: np.ndarray):
    '''
    将点从坐标系w，变换到像素坐标系
    Args:
        T_cw: 外参矩阵 [4,4]
        K: 内参矩阵 [3,3]
        xyz_Pw: 点集合 [3,N]
    Returns: xy：[2,N]像素坐标（np.int16），depth:[N]深度值（np.float32）

    '''
    xyz_ci = T_cw[:3, :3].dot(xyz_Pw) + np.expand_dims(T_cw[:3, 3], axis=1)  # [3,N]
    xyd = K.dot(xyz_ci)  # [3,N]
    depth = xyd[2, :]  # [N]
    xy = (xyd / depth)[:2]  # [2,N]
    xy = xy.astype(np.int16)
    return xy, depth


def read_semantic_mask(mask_path: str, semantic_value: List[int] = None, dilate_size: int = -1,
                       img_size: Tuple[int, int] = (),
                       maps: Tuple[np.ndarray, np.ndarray] = None,
                       camera_mask: np.ndarray = None,
                       ):
    """
    读取语义mask，并进行膨胀处理
    Args:
        camera_mask: (w,h),相机mask
        maps: (un_map1,un_map2) 用于去畸变的映射矩阵
        img_size: (w,h)要缩放到图像的大小
        mask_path: mask图像的路径
        semantic_value: 语义值（mask掉的区域）
        dilate_size: 膨胀的半径

    Returns: 二值语义图像 [H,W]
    """
    if semantic_value is None:
        semantic_value = [20]

    mask_img = cv2.imread(mask_path, -1)

    if mask_img is None:
        mask_img = np.ones((img_size[1],img_size[0]),dtype=bool)
        return mask_img

    assert mask_img.ndim == 2

    # 提取特征语义值的mask
    all_semantic_mask = None
    if len(semantic_value) == 1:
        all_semantic_mask = (mask_img == semantic_value[0])  # 车辆的像素值为20
    else:
        all_semantic_mask = mask_img == semantic_value[0]
        for value in semantic_value[1:]:
            all_semantic_mask = np.logical_or(all_semantic_mask, mask_img == value)
    # 去除mask区域
    all_semantic_mask = ~all_semantic_mask

    if camera_mask is not None:
        assert mask_img.shape[0] == camera_mask.shape[0]
        assert mask_img.shape[1] == camera_mask.shape[1]
        all_semantic_mask = np.logical_and(all_semantic_mask,camera_mask)

    mask_img = all_semantic_mask.astype(np.uint8)

    # 去畸变
    if maps is not None:
        mask_img = cv2.remap(mask_img, maps[0], maps[1], cv2.INTER_NEAREST)

    # 腐蚀
    if dilate_size > 0:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        mask_img = cv2.erode(mask_img, kernel, iterations=1)

    # 缩放
    if img_size:
        mask_img = cv2.resize(mask_img, img_size, interpolation=cv2.INTER_NEAREST)
    return mask_img


class MaxieyeMVSDataset(Dataset):
    def __init__(
            self,
            data_root: str,  # 根目录
            lidar_data_root: str,  # lidar点云所在的文件夹
            num_views: int = 10,
            max_dim: int = -1,
            scan_list: List[str] = '',
            robust_train: bool = False,
            regenerate_depth_image: bool = True,
            vis_sample: bool = False,  # 是否可视化图像
            use_srcs_pointcloud=True,  # 是否使用多帧点云叠在一起
            camera_mask_path = '',
    ) -> None:
        super(MaxieyeMVSDataset, self).__init__()

        self.regenerate_depth_image = regenerate_depth_image
        self.data_root = data_root
        self.num_views = num_views
        self.max_dim = max_dim
        self.robust_train = robust_train
        self.metas: List[Tuple[str, str, int, List[int]]] = []
        self.vis_sample = vis_sample
        self.use_srcs_pointcloud = use_srcs_pointcloud

        self.colmap_intrinsic: Dict[int, np.ndarray] = {}
        self.colmap_extrinsic: Dict[str, np.ndarray] = {}
        self.colmap_images_points: Dict[str, List[np.ndarray]] = {}

        self.train_list = [data_root + '/' + dataset for dataset in scan_list]
        self.train_lidar_list = [lidar_data_root + '/' + dataset for dataset in scan_list]

        param_infos_list, data_infos_list, self.image_paths_list, self.data_idxs = get_data_param(self.train_list)

        self.K_ori_list = []  # 原图的内参
        self.rot_list = []  # 外参 R_car_cam，或 R_cam2car
        self.tran_list = []  # 外参 t_car_cam，或 R_cam2car
        self.T_lidar2cam_list = []
        self.distorts_list = []  # 畸变系数
        self.H_list = []  # 原始图像的高
        self.W_list = []
        self.ixes = []

        self.idx_map = {}

        self.camera_mask = None
        if camera_mask_path != '':
            self.camera_mask = cv2.imread(camera_mask_path, 0)

        for param_infos in param_infos_list:  # 遍历bag
            self.H_list.append(param_infos['imgH_ori'])
            self.W_list.append(param_infos['imgW_ori'])
            self.K_ori_list.append(param_infos['ori_K'])  # 内参
            self.distorts_list.append(np.array(param_infos['dist_coeffs']))
            self.T_lidar2cam_list.append(convert_rvector_to_pose_mat(param_infos['rvector']))  # 旋转向量（lidar2cam）
            # 外参
            yaw, pitch, roll = param_infos['yaw'], param_infos['pitch'], param_infos['roll']
            self.rot_list.append(convert_rollyawpitch_to_rot(roll, yaw, pitch).I)
            self.tran_list.append(np.array(param_infos['xyz']))

        sample_list = {}  # 保存了每个图像的位姿、内参等信息
        for i in range(len(self.image_paths_list)):  # 遍历bag
            # 获取该bag的参数
            data_infos = data_infos_list[i]
            lidar2cam_vector = self.T_lidar2cam_list[i]
            image_paths = self.image_paths_list[i]
            rot, tran = self.rot_list[i], self.tran_list[i]
            K_ori = self.K_ori_list[i]
            distorts = self.distorts_list[i]
            H, W = self.H_list[i], self.W_list[i]
            # 该bag下所有图像的路径
            image_list = sorted(glob.glob(image_paths))

            sub_samples = []
            end_idx = len(data_infos)

            # 遍历所有图像，得到其位姿
            for j in range(0, end_idx, 1):
                img_path = image_list[j]
                img_name = os.path.split(img_path)[-1]
                ipose = data_infos[img_name]['ipose']
                pose = convert_quat_to_pose_mat(ipose[1:8])  # 图像位姿
                sub_samples.append((img_path, pose))

            sample_list[image_paths] = [sub_samples, rot, tran, K_ori, distorts, H, W, lidar2cam_vector]

        self.ixes = sample_list

        # 去畸变
        self.K_un_list = []
        self.un_map1_list = []
        self.un_map2_list = []

        for sce_id in range(len(self.image_paths_list)):  # 遍历bag
            sce_name = self.image_paths_list[sce_id]

            K_ori = sample_list[sce_name][3]
            distorts = sample_list[sce_name][4]
            H = sample_list[sce_name][5]
            W = sample_list[sce_name][6]

            # 计算去畸变映射
            K_ori = np.array(K_ori).reshape(3, 3)
            whole_img_size = (W, H)
            alpha = 0
            # getOptimalNewCameraMatrix的输入参数为1、原始相机内参，2、畸变参数，3、图像原始尺寸，4、alpha值，5、去畸后的图片尺寸，6、centerPrincipalPoint，
            K_un, _ = cv2.getOptimalNewCameraMatrix(K_ori, distorts, whole_img_size, alpha, whole_img_size)
            # 指定去畸变前后图像尺寸相同，此处可以调整修改，影响不大；
            map1, map2 = cv2.initUndistortRectifyMap(K_ori, distorts, None, K_un, whole_img_size, cv2.CV_32FC1)

            self.K_un_list.append(K_un.astype(np.float32))
            self.un_map1_list.append(map1)
            self.un_map2_list.append(map2)

    def __len__(self):
        return len(self.data_idxs)

    def get_ref_srcs(self, ref_idx):
        """
        分配参考视图和源视图
        Args:
        Returns: 返回匹配关系，和每个图像的位姿
        """
        ref_sce_id, ref_sce_id_ind = self.data_idxs[ref_idx]
        sce_name = self.image_paths_list[ref_sce_id]

        srcs_idx_list = []  # 配对的源视图索引
        num_of_src = self.num_views - 1

        for i in range(1, num_of_src + 1):
            src_idx = ref_idx + i
            if src_idx < len(self.data_idxs):
                src_sce_id, src_sce_id_ind = self.data_idxs[src_idx]
                if src_sce_id == ref_sce_id:
                    srcs_idx_list.append(src_idx)
            if len(srcs_idx_list) >= num_of_src:
                break
            src_idx = ref_idx - i
            if src_idx >= 0:
                src_sce_id, src_sce_id_ind = self.data_idxs[src_idx]
                if src_sce_id == ref_sce_id:
                    srcs_idx_list.append(src_idx)
            if len(srcs_idx_list) >= num_of_src:
                break

        poses_dict: Dict[type(ref_idx), Dict[str, np.matrix]] = {}  # 参考视图和源视图的位姿

        T_lidar2cam = self.T_lidar2cam_list[ref_sce_id]
        T_lc = np.linalg.inv(T_lidar2cam)

        image_path, T_wl = self.ixes[sce_name][0][ref_sce_id_ind]
        T_wl = T_wl.A.copy()  # 转换为narray格式

        xyz_offset = T_wl[:3, 3].copy()
        T_wl[:3, 3] = T_wl[:3, 3] - xyz_offset  # 平移量减去同一个值，防止量化误差
        T_wc = T_wl.dot(T_lc)

        poses_dict[ref_idx] = {}
        poses_dict[ref_idx]["Twl"] = T_wl
        poses_dict[ref_idx]["Twc"] = T_wc

        self.idx_map[ref_idx] = image_path

        # 输出到txt
        f = open(os.path.join(self.data_root,"pair.txt"),'a')
        f.write(str(ref_idx)+'\n')
        f.write(str(len(srcs_idx_list))+" ")

        for src_idx in srcs_idx_list:
            src_sce_id, src_sce_id_ind = self.data_idxs[src_idx]
            image_path, T_wl = self.ixes[sce_name][0][src_sce_id_ind]
            T_wl = T_wl.A.copy()
            # print(f"{src_idx} {T_wl}")

            T_wl[:3, 3] = T_wl[:3, 3] - xyz_offset  # 平移量减去同一个值，防止量化误差
            T_wc = T_wl.dot(T_lc)

            poses_dict[src_idx] = {}
            poses_dict[src_idx]["Twl"] = T_wl
            poses_dict[src_idx]["Twc"] = T_wc

            self.idx_map[src_idx] = image_path
            score = abs(src_idx - ref_idx)
            f.write(str(src_idx)+" "+str(score)+" ")
        f.write('\n')
        f.close()

        return srcs_idx_list, poses_dict

    def write_idx_map(self,write_name="map.txt"):
        with open(os.path.join(self.data_root,write_name), 'w') as write_f:
            write_f.write(json.dumps(self.idx_map, indent=4, ensure_ascii=False))

    def __getitem__(self, idx):
        sce_id, sce_id_ind = self.data_idxs[idx]  # 获得bag索引和图像索引
        sce_name = self.image_paths_list[sce_id]

        # 参考视图-源视图的配对
        srcs_idx_list, poses_dict = self.get_ref_srcs(idx)
        view_ids = [idx] + srcs_idx_list

        assert len(view_ids) == self.num_views

        images = []
        intrinsics_list = []
        extrinsics_list = []
        depth_min: float = 2
        depth_max: float = 100

        # 获取内参
        K_un = self.K_un_list[sce_id]
        un_map1, un_map2 = self.un_map1_list[sce_id], self.un_map2_list[sce_id]

        ref_image = None  # 用于可视化

        for i, image_idx in enumerate(view_ids):
            image_sce_id, image_sce_id_ind = self.data_idxs[image_idx]
            # 图像去畸变
            image_path, _ = self.ixes[sce_name][0][image_sce_id_ind]
            img = cv2.imread(image_path)
            img_un = cv2.remap(img, un_map1, un_map2, cv2.INTER_LINEAR)
            # 缩放图像
            img_un, original_h, original_w = scale_to_max_dim(img_un, self.max_dim)
            if i == 0:
                ref_image = img_un
            img_un = np.array(img_un, dtype=np.float32) / 255.0

            # 根据图像，缩放内参
            K = K_un.copy()  # [3,3]
            K[0] *= img_un.shape[1] / original_w  # fx,cx缩放
            K[1] *= img_un.shape[0] / original_h
            intrinsics_list.append(K)

            images.append(img_un.transpose([2, 0, 1]))

            T_wc = poses_dict[image_idx]["Twc"].A.astype(np.float32)
            # if abs(T_wc.max()) > 1000:
            #    print(i, image_idx)
            # print(T_wc)
            extrinsics_list.append(T_wc)

        intrinsics = np.stack(intrinsics_list)  # [N,3,3]
        extrinsics = np.stack(extrinsics_list)  # [N,4,4]

        H, W, C = ref_image.shape

        T_cl = self.T_lidar2cam_list[sce_id].A  # 外参
        depth_img = np.empty(0)  # 深度图像

        ref_image_path, _ = self.ixes[sce_name][0][sce_id_ind]
        ref_depth_path = os.path.join(self.train_lidar_list[sce_id], "depths", Path(ref_image_path).stem + ".png")

        if not self.regenerate_depth_image and os.path.exists(ref_depth_path):
            depth_img = cv2.imread(ref_depth_path, -1)
        else:
            ref_semantic_mask = None  # 参考视图的语义mask
            depth_img = np.zeros((H, W), dtype=np.float32)

            T_wl0 = poses_dict[view_ids[0]]["Twl"]  # 获得参考帧的位姿
            K0 = intrinsics_list[0]  # 参考帧的内参

            # 将所有的点云融合到参考视图
            for i, image_idx in enumerate(view_ids):
                image_sce_id, image_sce_id_ind = self.data_idxs[image_idx]
                image_path, _ = self.ixes[sce_name][0][image_sce_id_ind]
                T_wli = poses_dict[image_idx]["Twl"]
                T_l0li = np.linalg.inv(T_wl0).dot(T_wli)  # 从lidar_i到lidar_0的相对变换
                Ki = intrinsics_list[i]  # 源视图的内参

                # 读取点云
                timestamp = Path(image_path).stem + ".npy"
                point_path = os.path.join(self.train_lidar_list[sce_id], "depths_single_with_obj_time_ori", timestamp)
                points = np.load(point_path)  # [N,5]
                xyz_li = points[:, :3]  # [N,3]
                xyz_li = xyz_li.T  # [3,N]

                # 读取语义mask
                timestamp = Path(image_path).stem + ".png"
                mask_path = os.path.join(self.train_lidar_list[sce_id], "semantic_maps_ori", timestamp)
                semantic_mask_img = read_semantic_mask(mask_path, img_size=(W, H), maps=(un_map1, un_map2),
                                                       dilate_size=15, semantic_value=[20],
                                                       camera_mask=self.camera_mask)

                if i == 0:
                    ref_semantic_mask = semantic_mask_img
                else:  # 根据语义分割的mask，去除动态物体
                    xy, depth = project_to_pixel(T_cl, Ki, xyz_li)  # 首先将点投影到相机坐标系i

                    # 处于图像之外的点
                    field_mask1 = np.logical_or(xy[0, :] <= 0, xy[0, :] >= W)
                    field_mask2 = np.logical_or(xy[1, :] <= 0, xy[1, :] >= H)
                    field_mask = np.logical_or(field_mask1, field_mask2)

                    xx, yy = xy[0], xy[1]
                    xx = np.clip(xx, 0, W - 1)
                    yy = np.clip(yy, 0, H - 1)

                    semantic_mask = semantic_mask_img[yy, xx] > 0
                    depth_mask = np.logical_and(depth >= depth_min, depth <= depth_max)
                    # 只保留非动态区域，且深度值有效的点
                    semantic_mask = np.logical_and(semantic_mask, depth_mask)

                    final_mask = np.logical_or(field_mask, semantic_mask)
                    xyz_li = xyz_li[:, final_mask]

                # 变换到lidar0坐标系
                xyz_l0 = T_l0li[:3, :3].dot(xyz_li) + np.expand_dims(T_l0li[:3, 3], axis=1)
                xy, depth = project_to_pixel(T_cl, K0, xyz_l0)  # 投影到像素坐标系

                # 只判断图像区域内的mask
                field_mask1 = np.logical_and(xy[0, :] >= 0, xy[0, :] < W)
                field_mask2 = np.logical_and(xy[1, :] >= 0, xy[1, :] < H)
                field_mask = np.logical_and(field_mask1, field_mask2)

                # 只保留投影后非动态物体的点
                xx, yy = xy[0], xy[1]
                xx = np.clip(xx, 0, W - 1)
                yy = np.clip(yy, 0, H - 1)
                semantic_mask = ref_semantic_mask[yy, xx] > 0

                # 只保留有效深度内的点
                depth_mask = np.logical_and(depth >= depth_min, depth <= depth_max)

                final_mask = np.logical_and(field_mask, semantic_mask)
                final_mask = np.logical_and(final_mask, depth_mask)

                xy = xy[:, final_mask]  # [2,N]
                depth = depth[final_mask]
                if len(depth) == 0:
                    continue

                depth_img[xy[1, :], xy[0, :]] = depth

                if self.vis_sample:
                    for i in range(xy.shape[1]):
                        cv2.circle(ref_image, (xy[0, i], xy[1, i]), 2, (255, 0, 255), -1)

                if not self.use_srcs_pointcloud:
                    break

                # cv2.imwrite(depth_path,depth_img)
                # print("write to:%s" % depth_path)

        depth_gt = np.expand_dims(depth_img, 0)
        mask = depth_gt >= depth_min

        if self.vis_sample:
            cv2.imshow("ref_img", ref_image)
            cv2.waitKey(50)

        # for i in range(len(images)):
        #    print("dataloader images",images[i].shape)
        # print("dataloader images_num:%d" % (len(images)))
        # print("dataloader intrinsics",intrinsics.shape)
        # print("dataloader extrinsics",extrinsics.shape)
        # print("dataloader depth_gt",depth_gt.shape)
        # print("dataloader mask",mask.shape)

        return {
            "images": images,  # List[Tensor]: [N][3,Hi,Wi], N is number of images
            "intrinsics": intrinsics,  # Tensor: [N,3,3]
            "extrinsics": extrinsics,  # Tensor: [N,4,4]
            "depth_min": depth_min,  # Tensor: [1]
            "depth_max": depth_max,  # Tensor: [1]
            "depth_gt": depth_gt,  # Tensor: [1,H0,W0] if exists
            "mask": mask,
            "file_path":self.idx_map[idx]
        }


if __name__ == "__main__":
    data_root = "/home/cjq/data/mvs/kaijin"
    lidar_data_root = "/home/cjq/data/mvs/lidar"

    train_datasets = ['20221027_6.58km_2022-12-04-09-08-14', ]

    dataloader = MaxieyeMVSDataset(data_root, lidar_data_root,
                                   num_views=7,
                                   max_dim=800,
                                   scan_list=train_datasets,
                                   vis_sample=True,
                                   use_srcs_pointcloud=True,
                                   camera_mask_path='/media/cjq/新加卷/datasets/220Dataset/GS4-2M.png'
                                   )

    image_loader = DataLoader(
        dataset=dataloader, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

    cnt=0

    for batch_idx, sample in enumerate(image_loader):
        start_time = time.time()

        # sample_cuda = to_cuda(sample)
        # print(type(sample_cuda["images"]), sample_cuda["images"][0].shape, sample_cuda["images"][0].dtype)
        # print("depth_gt.shape", sample_cuda["depth_gt"].shape)
        # print("depth_gt.mask", sample_cuda["mask"].shape)

        ref_image = sample["images"][0][0]  # [3,H,W]
        depth_image = sample["depth_gt"][0]  # [1,H,W]
        K = sample["intrinsics"][0][0]  # [3,3]

        cnt +=1

        if cnt>10:
            break

        # print(type(ref_image))
        # print(ref_image.shape)
        # print(type(depth_image))
        # print(depth_image.shape)

        #if batch_idx%20==0:
            # xyz, rgb = generate_pointcloud(depth_image[0].numpy(), ref_image.numpy(), K.numpy())
            # xyz:[3,N]
            # rgb:[N,3]
            #vis_points([xyz.transpose()],[rgb])

        xyz, rgb = generate_pointcloud(depth_image[0].numpy(), ref_image.numpy(), K.numpy())
        # xyz:[3,N]
        # rgb:[N,3]
        # vis_points([xyz.transpose()],[rgb])
        write_ply("test.ply",xyz.transpose(),rgb)

    dataloader.write_idx_map()

