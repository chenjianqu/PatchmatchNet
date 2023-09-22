import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from pose_utils import convert_rollyawpitch_to_rot, convert_rvector_to_pose_mat, convert_quat_to_pose_mat


def get_pose_mat(sce_id, recs, extrin_rot, extrin_tran, T_lidar2cam_list):
    '''
    获取图像 car2world的位姿
    Args:
        sce_id: 数据集的索引
        recs: 图像路径
        extrin_rot: 外参 R_car_cam，或 R_cam2car
        extrin_tran:
    Returns: 返回每个图像的位姿和相对位姿，其中位姿 [len(recs), 3, 4],位姿是car的位姿 T_car2world或Twc
    '''
    T_lidar2cam = T_lidar2cam_list[sce_id]
    pose_mats = np.zeros((len(recs), 3, 4))
    pose_deltas = np.zeros((len(recs), 3, 4))
    for ii, sample in enumerate(recs):
        image_path, pose = sample  # 其中 pose是lidar的位姿， T_wl
        rot_lidar2car = extrin_rot.dot(T_lidar2cam[:3, :3])  # R_car_lidar = R_car_cam * R_cam_lidar
        tran_lidar2car = extrin_rot.dot(T_lidar2cam[:3, 3]).T + extrin_tran
        rot_car2world = pose[:3, :3].dot(rot_lidar2car.getI())
        tran_car2world = pose[:3, :3].dot(tran_lidar2car.T) + pose[:3, 3]
        pose_mat = np.eye(4, dtype=np.float64)
        pose_mat[:3, :3] = rot_car2world
        pose_mat[:3, 3] = tran_car2world.T
        pose_mat = np.matrix(pose_mat)
        pose_mats[ii, :, :] = pose_mat[:3, :]

        # 相对位姿
        if ii > 0:
            pose_diff = pose_prev.getI().dot(pose_mat)
            pose_deltas[ii, :, :] = pose_diff[:3, :]
        pose_prev = pose_mat
    return torch.Tensor(pose_deltas), torch.DoubleTensor(pose_mats)


def get_ref_srcs(sce_id, recs, T_lidar2cam_list):
    '''
    分配参考视图和源视图
    Args:
        sce_id: 数据集的索引
        recs: 图像路径
    Returns: 返回每个图像的位姿和相对位姿，其中位姿 [len(recs), 3, 4],位姿是car的位姿 T_car2world或Twc
    '''
    T_lidar2cam = T_lidar2cam_list[sce_id]
    T_lc = np.linalg.inv(T_lidar2cam)

    num_image = len(recs)

    # 构建参考-源视图的关系
    score = np.zeros((num_image, num_image))
    queue = []
    for i in range(num_image):
        for j in range(i + 1, num_image):
            queue.append((i, j))
    for i, j in queue:
        s = num_image - abs(i - j)  # 这里的假设是：图像已经按照时间进行排序
        score[i, j] = s
        score[j, i] = s
    view_sel = []
    for i in range(num_image):
        sorted_score = np.argsort(score[i])[::-1]
        view_sel.append([(k, score[i, k]) for k in sorted_score[:10]])

    # 计算外参
    poses_cam = []
    for idx in range(num_image):
        image_path, T_wl = recs[idx]
        T_wc = T_wl.dot(T_lc)
        poses_cam.append(T_wc)

    return view_sel, poses_cam


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


def generate_depth_map(ref_image_path, src_images_path, K, poses):
    pass


if __name__ == "__main__":
    use_train = True
    data_root = "/home/cjq/data/mvs/kaijin"
    lidar_data_root = "/home/cjq/data/mvs/lidar"

    train_datasets = ['20221020_5.78km_2022-11-29-13-55-40', ]
    val_datasets = ['20221020_5.78km_2022-11-29-13-55-40']

    if not use_train:
        train_datasets = val_datasets

    train_list = [data_root + '/' + dataset for dataset in train_datasets]
    train_lidar_list = [lidar_data_root + '/' + dataset for dataset in train_datasets]

    param_infos_list, data_infos_list, image_paths_list, idxs = get_data_param(train_list)

    K_ori_list = []
    rot_list = []
    tran_list = []
    T_lidar2cam_list = []
    distorts_list = []
    H_list = []
    W_list = []
    # 遍历bag
    for param_infos in param_infos_list:
        H_list.append(param_infos['imgH_ori'])
        W_list.append(param_infos['imgW_ori'])

        K_ori_list.append(param_infos['ori_K'])  # 内参
        dist_coeffs = param_infos['dist_coeffs']  # 畸变系数
        distorts_list.append(np.array(dist_coeffs))

        rvector = param_infos['rvector']  # 旋转向量（lidar2cam）
        T_lidar2cam_list.append(convert_rvector_to_pose_mat(rvector))

        # 外参
        yaw, pitch, roll = param_infos['yaw'], param_infos['pitch'], param_infos['roll']
        rot_list.append(convert_rollyawpitch_to_rot(roll, yaw, pitch).I)
        tran_list.append(np.array(param_infos['xyz']))

    sample_list = {}
    # 遍历bag
    for i in range(len(image_paths_list)):
        # 获取该bag的参数
        data_infos = data_infos_list[i]
        lidar2cam_vector = T_lidar2cam_list[i]
        image_paths = image_paths_list[i]
        rot = rot_list[i]
        tran = tran_list[i]
        K_ori = K_ori_list[i]
        H = H_list[i]
        W = W_list[i]
        distorts = distorts_list[i]
        # 该bag下所有图像的路径
        image_list = sorted(glob.glob(image_paths))

        sub_samples = []
        end_idx = len(data_infos)

        # 遍历所有图像，得到其位姿
        for j in range(0, end_idx, 1):
            imgname = os.path.split(image_list[j])[-1]
            ipose = data_infos[imgname]['ipose']
            pose = convert_quat_to_pose_mat(ipose[1:8])  # 图像位姿
            sub_samples.append((image_list[j], pose))

        sample_list[image_paths] = [sub_samples, rot, tran, K_ori, distorts, H, W, lidar2cam_vector]

    sce_id = 0
    sce_name = image_paths_list[sce_id]
    lidar_path = train_lidar_list[sce_id]

    sce_id_ind = 0

    # 获取范围内的图像路径
    stride = 1
    bag_len = len(sample_list[sce_name][0])

    # recs = [sample_list[sce_name][0][ii] for ii in range(len(sample_list[sce_name][0]))]
    recs = [sample_list[sce_name][0][ii] for ii in range(1000)]

    rot = sample_list[sce_name][1]
    tran = sample_list[sce_name][2]
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

    # pose_deltas, pose_mats = get_pose_mat(sce_id, recs, rot, tran,T_lidar2cam_list)

    view_sel, poses_cam = get_ref_srcs(sce_id, recs, T_lidar2cam_list)

    T_lidar2cam = T_lidar2cam_list[sce_id]

    for ref_idx, srcs in enumerate(view_sel):
        ref_image_path, _ = recs[ref_idx]
        ref_img = cv2.imread(ref_image_path)

        ref_img_un = cv2.remap(ref_img, map1, map2, cv2.INTER_LINEAR)

        timestamp = Path(ref_image_path).stem + ".npy"
        point_path = os.path.join(lidar_path, "depths_single_with_obj_time_ori", timestamp)

        points = np.load(point_path)  # [N,5]
        xyz_lidar = points[:, :3]  # [N,3]
        xyz_lidar = xyz_lidar.T  # [3,N]
        xyz_cam = T_lidar2cam[:3, :3].dot(xyz_lidar) + T_lidar2cam[:3, 3]  # [3,N]

        xyd = K_un.dot(xyz_cam)  # [3,N]
        xy = (xyd / xyd[2, :])[:2]  # [2,N]
        depth = xyd[2, :]

        num_points = xy.shape[1]
        for i in range(0, num_points):
            center = (int(xy[0, i]), int(xy[1, i]))
            cv2.circle(ref_img_un, center, 2, (255, 0, 255), -1)

        cv2.imshow("ref_img", ref_img_un)
        cv2.waitKey(50)

