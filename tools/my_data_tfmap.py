"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import numpy as np
import cv2
from glob import glob
import json
import glob
import math
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data.distributed import DistributedSampler

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

vis_depth = False
pi = 3.1415926


def euler_angles_to_rotation_matrix(angle, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    roll, pitch, yaw = angle[0], angle[1], angle[2]

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]])

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = np.array(R)

    if is_dir:
        R = R[:, 2]

    return R


def convert_quat_to_pose_mat(xyzquat):
    xyz = xyzquat[:3]
    quat = xyzquat[3:]
    ret = R.from_quat(quat)
    # print('xxx = ', xyz, quat, Rot)
    T = np.array(xyz).reshape(3, 1)
    pose_mat = np.eye(4, dtype=np.float64)
    pose_mat[:3, :3] = ret.as_matrix()
    pose_mat[:3, 3] = T.T
    return np.matrix(pose_mat)


def convert_6dof_to_pose_mat(dof6):
    xyz = dof6[:3]
    angle = dof6[3:]
    R = euler_angles_to_rotation_matrix(angle)
    T = np.array(xyz).reshape(3, 1)
    pose_mat = np.eye(4, dtype=np.float64)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = T.T
    return np.matrix(pose_mat)


def convert_rvector_to_pose_mat(dof6):
    xyz = dof6[:3]
    angle = dof6[3:]
    # R = euler_angles_to_rotation_matrix(angle)
    R, _ = cv2.Rodrigues(np.array(angle))
    T = np.array(xyz).reshape(3, 1)
    pose_mat = np.eye(4, dtype=np.float64)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = T.T
    return np.matrix(pose_mat)


def convert_rollyawpitch_to_pose_mat(roll, yaw, pitch, x, y, z):
    roll *= pi / 180.
    yaw *= pi / 180.
    pitch *= pi / 180.
    Rr = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(roll), np.sin(roll)],
                   [0.0, -np.sin(roll), np.cos(roll)]], dtype=np.float32)
    Ry = np.array([[np.cos(pitch), 0.0, -np.sin(pitch)],
                   [0.0, 1.0, 0.0],
                   [np.sin(pitch), 0.0, np.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[np.cos(yaw), np.sin(yaw), 0.0],
                   [-np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.matrix(Rr) * np.matrix(Rx) * np.matrix(Ry) * np.matrix(Rz)
    T = np.array([x, y, z]).reshape(3, 1)
    pose_mat = np.eye(4, dtype=np.float64)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = T.T
    return np.matrix(pose_mat)


def convert_rollyawpitch_to_rot(roll, yaw, pitch):
    roll *= pi / 180.
    yaw *= pi / 180.
    pitch *= pi / 180.
    Rr = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(roll), np.sin(roll)],
                   [0.0, -np.sin(roll), np.cos(roll)]], dtype=np.float32)
    Ry = np.array([[np.cos(pitch), 0.0, -np.sin(pitch)],
                   [0.0, 1.0, 0.0],
                   [np.sin(pitch), 0.0, np.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[np.cos(yaw), np.sin(yaw), 0.0],
                   [-np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.matrix(Rr) * np.matrix(Rx) * np.matrix(Ry) * np.matrix(Rz)
    return R


class TFmapData(torch.utils.data.Dataset):
    def __init__(self, data_idxs, param_infos_list, data_infos_list, vp_infos_list,
                 image_paths_list, las_paths_list, seg_paths_list, is_train, seq_len,
                 used_fcodes, sample_num, crop_size):
        self.data_idxs = data_idxs
        self.data_infos_list = data_infos_list
        self.vp_infos_list = vp_infos_list
        self.is_train = is_train
        self.seq_len = seq_len
        self.used_fcodes = used_fcodes
        self.crop_size = crop_size
        self.image_paths_list = image_paths_list

        self.las_paths_list = las_paths_list
        self.seg_paths_list = seg_paths_list
        self.sample_num = sample_num

        # K_list = []
        K_ori_list = []
        rot_list = []
        tran_list = []
        T_lidar2cam_list = []
        distorts_list = []
        H_list = []
        W_list = []

        for param_infos in param_infos_list:
            # print (param_infos)
            # K_list.append(param_infos['K'])
            K_ori_list.append(param_infos['ori_K'])  # 内参
            H_list.append(param_infos['imgH_ori'])
            W_list.append(param_infos['imgW_ori'])
            rvector = param_infos['rvector']  # 旋转向量（lidar2cam）
            T_lidar2cam_list.append(convert_rvector_to_pose_mat(rvector))
            yaw = param_infos['yaw']
            pitch = param_infos['pitch']
            roll = param_infos['roll']
            dist_coeffs = param_infos['dist_coeffs']  # 畸变系数
            distorts_list.append(np.array(dist_coeffs))
            rot_list.append(convert_rollyawpitch_to_rot(roll, yaw, pitch).I)
            tran = param_infos['xyz']
            tran_list.append(np.array(tran))

        # self.K_list = K_list
        self.rot_list = rot_list
        self.tran_list = tran_list
        self.T_lidar2cam_list = T_lidar2cam_list
        self.distorts_list = distorts_list
        # self.maps_list, self.ignore_maps_list, self.ob_maps_list  = self.get_maps()
        self.maps_list, self.ignore_maps_list, self.ob_maps_list = [], [], []
        self.K_ori_list = K_ori_list
        self.H_list = H_list
        self.W_list = W_list
        self.ixes = self.prepro()  # 所有位姿的内参、外参，位姿

    def prepro(self):
        sample_list = {}
        for i in range(len(self.image_paths_list)):
            # if i == 0:
            #     continue
            sub_samples = []
            # start_idx, end_idx = self.scenes[i]
            data_infos = self.data_infos_list[i]
            vp_infos = self.vp_infos_list[i]
            lidar2cam_vector = self.T_lidar2cam_list[i]
            image_paths = self.image_paths_list[i]
            image_list = sorted(glob.glob(image_paths))  # 该数据集下所有图像的路径
            rot = self.rot_list[i]
            tran = self.tran_list[i]
            # K = self.K_list[i]
            K_ori = self.K_ori_list[i]
            H = self.H_list[i]
            W = self.W_list[i]
            distorts = self.distorts_list[i]
            # end_idx = len(data_infos) - self.seq_len + 1
            end_idx = len(data_infos)
            for j in range(0, end_idx, 1):
                imgname = os.path.split(image_list[j])[-1]
                # print(image_list[i], imgname)
                ipose = data_infos[imgname]['ipose']
                pose = convert_quat_to_pose_mat(ipose[1:8])  # 图像位姿
                vp = vp_infos[imgname]['vp_pose']

                sub_samples.append((image_list[j], pose, vp))
                # sample_list.append([[0, end_idx - self.seq_len + 1], sub_samples, rot, tran])
            sample_list[image_paths] = [sub_samples, rot, tran, K_ori, distorts, H, W, lidar2cam_vector]
        return sample_list

    # 结构： {bag_name:[ sub_samples[[image_list[j], pose, vp],], rot, tran, K_ori, distorts, H, W, lidar2cam_vector],}


    def get_pose_mat(self, sce_id, recs, extrin_rot, extrin_tran):
        '''
        获取图像 car2world的位姿
        Args:
            sce_id:
            recs:
            extrin_rot:
            extrin_tran:
        Returns:
        '''
        T_lidar2cam = self.T_lidar2cam_list[sce_id]
        pose_mats = np.zeros((len(recs), 3, 4))
        pose_deltas = np.zeros((len(recs), 3, 4))
        pose_prev = None

        for ii, sample in enumerate(recs):
            image_path, pose, _ = sample
            rot_lidar2car = extrin_rot.dot(T_lidar2cam[:3, :3])
            tran_lidar2car = extrin_rot.dot(T_lidar2cam[:3, 3]).T + extrin_tran
            rot_car2world = pose[:3, :3].dot(rot_lidar2car.getI())
            tran_car2world = pose[:3, :3].dot(tran_lidar2car.T) + pose[:3, 3]
            pose_mat = np.eye(4, dtype=np.float64)
            pose_mat[:3, :3] = rot_car2world
            pose_mat[:3, 3] = tran_car2world.T
            pose_mat = np.matrix(pose_mat)
            pose_mats[ii, :, :] = pose_mat[:3, :]
            if ii > 0:
                pose_diff = pose_prev.getI().dot(pose_mat)
                pose_deltas[ii, :, :] = pose_diff[:3, :]

            pose_prev = pose_mat
        return torch.Tensor(pose_deltas), torch.DoubleTensor(pose_mats)


    def __len__(self):
        return len(self.data_idxs)


class SegmentationData(TFmapData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
        self.buf = {}

    def __getitem__(self, index):
        # index = self.data_idxs[index]
        # index = 1161
        sce_id, sce_id_ind = self.data_idxs[index]
        sce_name = self.image_paths_list[sce_id]

        max_stride = min((len(self.ixes[sce_name][0]) - sce_id_ind) / self.seq_len, 5)

        stride = np.random.randint(1, max_stride + 1)

        if not self.is_train:
            stride = 3

        # recs = [sub_ixes[ii] for ii in range(index, index+self.seq_len*stride, stride)]
        # 结构： {bag_name:[ sub_samples[[image_path, pose, vp],], rot, tran, K_ori, distorts, H, W, lidar2cam_vector],}
        # 获取范围内的图像路径
        recs = [self.ixes[sce_name][0][ii] for ii in range(sce_id_ind, sce_id_ind + self.seq_len * stride, stride)]
        if check_depth_exist([p[0] for p in recs]) is False:
            stride = 1
            recs = [self.ixes[sce_name][0][ii] for ii in range(sce_id_ind, sce_id_ind + self.seq_len * stride, stride)]
        rot = self.ixes[sce_name][1]
        tran = self.ixes[sce_name][2]
        K_ori = self.ixes[sce_name][3]
        distorts = self.ixes[sce_name][4]
        H = self.ixes[sce_name][5]
        W = self.ixes[sce_name][6]
        lidar2cam = self.ixes[sce_name][-1]
        # runs
        axisangle_limits = [[-4. / 180. * np.pi, 4. / 180. * np.pi], [-4. / 180. * np.pi, 4. / 180. * np.pi],
                            [-4. / 180. * np.pi, 4. / 180. * np.pi]]
        # roll pitch yaw
        tran_limits = [[-2., 2.], [-1., 1.], [-1., 1.]]

        # runs1
        axisangle_noises = [np.random.uniform(*angle_limit) for angle_limit in axisangle_limits]
        tran_noises = [np.random.uniform(*tran_limit) for tran_limit in tran_limits]

        if not self.is_train:
            tran_noises = np.zeros_like(tran_noises)
            axisangle_noises = np.zeros_like(axisangle_noises)

        noise_rot = euler_angles_to_rotation_matrix(axisangle_noises)
        noise_tran = np.array(tran_noises)
        extrin_rot = noise_rot.dot(rot)
        extrin_tran = noise_rot.dot(tran).T + noise_tran

        # 读取位姿数据
        pose_deltas, pose_mats = self.get_pose_mat(sce_id, recs, rot, tran)


def worker_rnd_init(x):
    np.random.seed(13 + x)


def check_depth_exist(img_list):
    for img_path in img_list:
        depth_file = img_path.replace("org/img_ori", "depths_single_with_obj_time_ori"). \
            replace("/share/", "/share/occ_data/occ_data_ori/").replace(
            ".jpg", ".npy")
        if not os.path.exists(depth_file):
            return False
    return True


def get_data_param(path_idxs, seq_len, is_train=True):
    data_infos_list = []
    param_infos_list = []
    vp_infos_list = []
    image_paths_list = []
    las_paths_list = []
    seg_paths_list = []
    idxs = []
    for sce_id, path in enumerate(path_idxs):
        # 读取内参和位姿
        param_path = path + '/gen/param_infos.json'
        with open(param_path, 'r') as ff:
            param_infos = json.load(ff)
            param_infos_list.append(param_infos)

        image_paths = os.path.join(path + '/org/img_ori', "*.jpg")
        image_paths_list.append(image_paths)
        data_infos = {}
        with open(path + '/gen/bev_infos_roadmap_fork.json', 'r') as ff:
            data_infos = json.load(ff)
            data_infos_list.append(data_infos)

        vp_infos = {}
        with open(path + '/gen/vp_infos_ori.json', 'r') as ff:
            vp_infos = json.load(ff)
            vp_infos_list.append(vp_infos)

        las_paths = path + '/las_top/*.las'
        las_paths_list.append(las_paths)

        seg_paths = path + '/semantic_maps_ori/*.png'
        seg_paths_list.append(seg_paths)

        if is_train:
            with open(path + '/gen/train_fork20.lst', 'r') as ff:
                lines = ff.readlines()
                image_list = sorted(glob.glob(image_paths))
                img_name_first = None
                for image_path in image_list:
                    img_name = float(image_path.split('/')[-1][:-4])
                    if img_name_first is None:
                        img_name_first = img_name
                    elif img_name - img_name_first < 0.05:
                        print(f"img_name_diff:{img_name - img_name_first} image_path:{image_path}")
                for item in lines:
                    if check_depth_exist(image_list[int(item): int(item) + seq_len]) is True:
                        idxs += [[sce_id, int(item)]]
        else:
            idxs += [[sce_id, index] for index in range(len(data_infos) - seq_len)]
    return param_infos_list, data_infos_list, vp_infos_list, image_paths_list, las_paths_list, \
        seg_paths_list, idxs


def compile_data(dataroot, bsz, seq_len, sample_num, nworkers, use_train=True):
    train_datasets = ['20230216_2.11km_ZJ227_0301_4',
                      ]
    val_datasets = ['20230216_2.11km_ZJ227_0301_4']
    if not use_train:
        train_datasets = val_datasets

    train_list = [dataroot + '/' + dataset for dataset in train_datasets]
    val_list = [dataroot + '/' + dataset for dataset in val_datasets]
    used_fcodes = {}
    trainloader = None
    if use_train:
        # 读取参数信息
        param_infos_list, data_infos_list, vp_infos_list, image_paths_list, las_paths_list, \
            seg_paths_list, train_idxs = get_data_param(train_list, seq_len, is_train=True)

        traindata = SegmentationData(train_idxs, param_infos_list, data_infos_list, vp_infos_list, image_paths_list,
                                     las_paths_list, seg_paths_list,
                                     is_train=True, seq_len=seq_len, used_fcodes=used_fcodes,
                                     sample_num=sample_num,
                                      crop_size=96)

        trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                                  shuffle=False,
                                                  num_workers=nworkers,
                                                  drop_last=True,
                                                  worker_init_fn=worker_rnd_init,
                                                  pin_memory=False, prefetch_factor=32)

    param_infos_list, data_infos_list, vp_infos_list, image_paths_list, las_paths_list, \
        seg_paths_list, val_idxs = get_data_param(
        val_list, seq_len, is_train=False)

    valdata = SegmentationData(val_idxs, param_infos_list, data_infos_list, vp_infos_list,
                               image_paths_list, las_paths_list, seg_paths_list, is_train=False,
                               seq_len=seq_len, used_fcodes=used_fcodes, sample_num=sample_num,
                               crop_size=96)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True, persistent_workers=False)

    return trainloader, valloader


if __name__ == "__main__":
    trainloader, valloader = compile_data("/home/cjq/data",1,100,10,1)
    print(type(trainloader))
    for a in valloader:
        print(type(a))

