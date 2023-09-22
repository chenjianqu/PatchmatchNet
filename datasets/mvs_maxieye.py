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
from tools.colmap_utils import read_model_get_points
from tools.pose_utils import convert_rvector_to_pose_mat, convert_rollyawpitch_to_rot, convert_quat_to_pose_mat
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
            vis_sample: bool = False,
    ) -> None:
        super(MaxieyeMVSDataset, self).__init__()

        self.regenerate_depth_image = regenerate_depth_image
        self.data_root = data_root
        self.num_views = num_views
        self.max_dim = max_dim
        self.robust_train = robust_train
        self.metas: List[Tuple[str, str, int, List[int]]] = []
        self.vis_sample = vis_sample

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
        '''
        分配参考视图和源视图
        Args:
        Returns: 返回匹配关系，和每个图像的位姿，其中位姿 [len(recs), 3, 4],位姿是car的位姿 T_car2world或Twc
        '''
        ref_sce_id, ref_sce_id_ind = self.data_idxs[ref_idx]
        sce_name = self.image_paths_list[ref_sce_id]

        srcs_idx_list = []  # 配对的源视图索引

        for i in range(1, self.num_views + 1):
            src_idx = ref_idx + i
            if src_idx < len(self.data_idxs):
                src_sce_id, src_sce_id_ind = self.data_idxs[src_idx]
                if src_sce_id == ref_sce_id:
                    srcs_idx_list.append(src_idx)
            if len(srcs_idx_list) >= self.num_views:
                break
            src_idx = ref_idx - i
            if src_idx >= 0:
                src_sce_id, src_sce_id_ind = self.data_idxs[src_idx]
                if src_sce_id == ref_sce_id:
                    srcs_idx_list.append(src_idx)
            if len(srcs_idx_list) >= self.num_views:
                break

        poses = {}  # 参考视图和源视图的位姿

        T_lidar2cam = self.T_lidar2cam_list[ref_sce_id]
        T_lc = np.linalg.inv(T_lidar2cam)

        image_path, T_wl = self.ixes[sce_name][0][ref_sce_id_ind]
        T_wc = T_wl.dot(T_lc)
        poses[ref_idx] = T_wc

        for src_idx in srcs_idx_list:
            src_sce_id, src_sce_id_ind = self.data_idxs[src_idx]
            image_path, T_wl = self.ixes[sce_name][0][src_sce_id_ind]
            T_wc = T_wl.dot(T_lc)
            poses[src_idx] = T_wc

        return srcs_idx_list, poses

    def __getitem__(self, idx):
        sce_id, sce_id_ind = self.data_idxs[idx]  # 获得bag索引和图像索引
        sce_name = self.image_paths_list[sce_id]

        srcs_idx_list, poses = self.get_ref_srcs(idx)
        assert len(srcs_idx_list) == self.num_views

        view_ids = [idx] + srcs_idx_list

        images = []
        intrinsics_list = []
        extrinsics_list = []
        depth_min: float = 2
        depth_max: float = 100

        # 获取内参
        K_un = self.K_un_list[sce_id]
        un_map1 = self.un_map1_list[sce_id]
        un_map2 = self.un_map2_list[sce_id]

        ref_image = None

        for i, image_idx in enumerate(view_ids):
            image_sce_id, image_sce_id_ind = self.data_idxs[image_idx]
            # 图像去畸变
            image_path, T_wl = self.ixes[sce_name][0][image_sce_id_ind]
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

            T_wc = poses[image_idx].A.astype(np.float32)
            extrinsics_list.append(T_wc)

        intrinsics = np.stack(intrinsics_list)  # [N,3,3]
        extrinsics = np.stack(extrinsics_list)  # [N,4,4]

        H, W, C = ref_image.shape

        T_cl = self.T_lidar2cam_list[sce_id].A  # 外参
        T_lc = np.linalg.inv(T_cl)

        depth_img = np.empty(0)

        ref_image_path, _ = self.ixes[sce_name][0][sce_id_ind]
        timestamp = Path(ref_image_path).stem + ".png"
        depth_path = os.path.join(self.train_lidar_list[sce_id], "depths", timestamp)

        if not self.regenerate_depth_image and os.path.exists(depth_path):
            depth_img = cv2.imread(depth_path, -1)
        else:
            depth_img = np.zeros((H, W), dtype=np.float32)
            # 将所有的点云融合到参考视图
            _, T_wl0 = self.ixes[sce_name][0][sce_id_ind]  # 获得参考帧的位姿
            K0 = intrinsics_list[0]  # 参考帧的内参
            for i, image_idx in enumerate(view_ids):
                image_sce_id, image_sce_id_ind = self.data_idxs[image_idx]
                image_path, T_wli = self.ixes[sce_name][0][image_sce_id_ind]
                T_l0li = np.linalg.inv(T_wl0).dot(T_wli)
                T_l0li = T_l0li.A  # 从lidar_i到lidar_0的相对变换
                Ki = intrinsics_list[i]  # 源视图的内参

                timestamp = Path(image_path).stem + ".npy"
                point_path = os.path.join(self.train_lidar_list[sce_id], "depths_single_with_obj_time_ori", timestamp)

                points = np.load(point_path)  # [N,5]
                xyz_li = points[:, :3]  # [N,3]
                xyz_li = xyz_li.T  # [3,N]

                # 根据语义分割的mask，去除动态物体

                # 首先将点投影到相机坐标系i
                xyz_ci = T_cl[:3, :3].dot(xyz_li) + np.expand_dims(T_cl[:3, 3], axis=1)  # [3,N]
                xyd = Ki.dot(xyz_ci)  # [3,N]
                depth = xyd[2, :]  # [N]
                xy = (xyd / depth)[:2]  # [2,N]
                xy = xy.astype(np.int16)

                # 处于图像之外的点
                field_mask = xy[0, :] <= 0
                field_mask = np.logical_and(field_mask, xy[0, :] >= W)
                field_mask = np.logical_and(field_mask, xy[1, :] <= 0)
                field_mask = np.logical_and(field_mask, xy[1, :] >= H)

                xx, yy = xy[0], xy[1]
                xx = np.clip(xx, 0, W - 1)
                yy = np.clip(yy, 0, H - 1)

                # 读取语义mask
                timestamp = Path(image_path).stem + ".png"
                mask_path = os.path.join(self.train_lidar_list[sce_id], "semantic_maps_ori", timestamp)
                mask_img = cv2.imread(mask_path, -1)
                mask_img = (mask_img == 20).astype(np.uint8)  # [N,] 车辆的像素值为20

                kernel = np.ones((4, 4), np.uint8)
                mask_img = cv2.dilate(mask_img, kernel, iterations=1)

                mask_value = mask_img[yy, xx]
                semantic_mask = mask_value == 0
                final_mask = np.logical_or(field_mask, semantic_mask)

                xyz_li = xyz_li[:, final_mask]

                # 变换到lidar0坐标系
                xyz_l0 = T_l0li[:3, :3].dot(xyz_li) + np.expand_dims(T_l0li[:3, 3], axis=1)

                xyz_c0 = T_cl[:3, :3].dot(xyz_l0) + np.expand_dims(T_cl[:3, 3], axis=1)  # [3,N]

                xyd = K0.dot(xyz_c0)  # [3,N]
                depth = xyd[2, :]  # [N]
                xy = (xyd / depth)[:2]  # [2,N]

                valid = xy[0, :] > 0
                valid = np.logical_and(valid, xy[0, :] < W)
                valid = np.logical_and(valid, xy[1, :] > 0)
                valid = np.logical_and(valid, xy[1, :] < H)

                xy = xy[:, valid]  # [2,N]
                depth = depth[valid]
                if len(depth) == 0:
                    continue

                xy = xy.astype(np.int16)
                depth_img[xy[1, :], xy[0, :]] = depth

                if self.vis_sample:
                    for i in range(xy.shape[1]):
                        cv2.circle(ref_image, (xy[0, i], xy[1, i]), 2, (255, 0, 255), -1)
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
        }


if __name__ == "__main__":
    data_root = "/home/cjq/data/mvs/kaijin"
    lidar_data_root = "/home/cjq/data/mvs/lidar"

    train_datasets = ['20221020_5.78km_2022-11-29-13-55-40', ]

    dataloader = MaxieyeMVSDataset(data_root, lidar_data_root,
                                   num_views=6,
                                   max_dim=640,
                                   scan_list=train_datasets)

    image_loader = DataLoader(
        dataset=dataloader, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    for batch_idx, sample in enumerate(image_loader):
        start_time = time.time()

        sample_cuda = to_cuda(sample)

        print(type(sample_cuda["images"]), sample_cuda["images"][0].shape, sample_cuda["images"][0].dtype)
        print("depth_gt.shape", sample_cuda["depth_gt"].shape)
        print("depth_gt.mask", sample_cuda["mask"].shape)
