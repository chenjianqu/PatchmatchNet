"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import os.path as Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from glob import glob
import time
import json
import glob
import math
import shapely
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import box
import re
import open3d as o3d
import torch
import shutil
import random
from scipy.spatial.transform import Rotation as R
from torch.utils.data.distributed import DistributedSampler
from src.tools import img_transform, normalize_img, gen_dx_bx, get_nusc_maps, get_local_map, get_rays, \
    get_rays_from_lidar, get_rays_from_lidar_points

from src.tools import get_ray_directions_ori as get_ray_directions
from src.tools import cvimg_transform_ori as cvimg_transform
from src.datasets.mask_utils import get_label_id_mapping
import copy
from kornia import create_meshgrid

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
    def __init__(self, data_idxs, param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list, roadmap_data_list,
                 roadmap_samples_list, roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list,
                 image_paths_list, las_paths_list, seg_paths_list, is_train, data_aug_conf, grid_conf, seq_len,
                 used_fcodes, sample_num, datatype, crop_size):
        self.data_idxs = data_idxs
        self.data_infos_list = data_infos_list
        self.vp_infos_list = vp_infos_list
        self.roadmap_data_list = roadmap_data_list
        self.roadmap_samples_list = roadmap_samples_list
        self.roadmap_forks_list = roadmap_forks_list
        self.mesh_objs_list = mesh_objs_list
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.seq_len = seq_len
        self.used_fcodes = used_fcodes
        self.crop_size = crop_size
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'], 4.0)
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        dx2, bx2, nx2 = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx2, self.bx2, self.nx2 = dx2.numpy(), bx2.numpy(), nx2.numpy()
        self.image_paths_list = image_paths_list
        self.map_paths_list = map_paths_list
        self.ignore_map_paths_list = ignore_map_paths_list
        self.ob_map_paths_list = ob_map_paths_list

        self.las_paths_list = las_paths_list
        self.seg_paths_list = seg_paths_list
        self.sample_num = sample_num
        self.datatype = datatype

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
            K_ori_list.append(param_infos['ori_K']) #内参
            H_list.append(param_infos['imgH_ori'])
            W_list.append(param_infos['imgW_ori'])
            rvector = param_infos['rvector'] # 旋转向量（lidar2cam）
            T_lidar2cam_list.append(convert_rvector_to_pose_mat(rvector))
            yaw = param_infos['yaw']
            pitch = param_infos['pitch']
            roll = param_infos['roll']
            dist_coeffs = param_infos['dist_coeffs'] #畸变系数
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
        self.ixes = self.prepro() # 所有位姿的内参、外参，位姿

    def get_maps(self):
        maps_list = []
        ignore_maps_list = []
        ob_maps_list = []
        for map_paths in self.map_paths_list:
            tf_maps = []
            for ii, path in enumerate(map_paths):
                with open(path, encoding='utf-8') as fp:
                    geojson = json.load(fp)
                    features = geojson['features']
                    map_data = []
                    for ii, feature in enumerate(features):
                        poly = feature['geometry']['coordinates']
                        type = feature['geometry']['type']
                        name = feature['properties']['name']
                        fcode = feature['properties']['FCode']
                        if not fcode in self.used_fcodes:
                            continue
                        if type == 'MultiLineString':
                            for pp in poly:
                                if len(pp) == 0:
                                    continue
                                data = np.array(pp, dtype=np.float64)
                                dist = np.sum(np.abs(data[-1] - data[0]))
                                label = self.used_fcodes[fcode]
                                if label == 6:
                                    map_data.append([data, 5])
                                elif name == '大面积车辆遮挡区域或路面不清晰' or fcode == '444300' or fcode == '401434' or fcode == '401453' or dist > 0.001:
                                    # print(path, name, fcode)
                                    map_data.append([data, -1])
                                else:
                                    if label == 3:
                                        map_data.append([data, 1])
                                    elif label == 4:
                                        map_data.append([data, 3])
                                    elif label == 5:
                                        map_data.append([data, 4])
                                    else:
                                        map_data.append([data, label])

                        elif type == "LineString":
                            if len(poly) == 0:
                                continue
                            data = np.array(poly, dtype=np.float64)
                            dist = np.sum(np.abs(data[-1] - data[0]))
                            label = self.used_fcodes[fcode]
                            if label == 6:
                                map_data.append([data, 5])
                            elif name == '大面积车辆遮挡区域或路面不清晰' or fcode == '444300' or fcode == '401434' or fcode == '401453' or dist > 0.001:
                                # print(path, name, fcode)
                                map_data.append([data, -1])
                            else:
                                if label == 3:
                                    map_data.append([data, 1])
                                elif label == 4:
                                    map_data.append([data, 3])
                                elif label == 5:
                                    map_data.append([data, 4])
                                else:
                                    map_data.append([data, label])
                        else:
                            print(type)
                    tf_maps += map_data
            maps_list.append(tf_maps)

        for ignore_map_paths in self.ignore_map_paths_list:
            tf_ignore_maps = []
            for ii, path in enumerate(ignore_map_paths):
                with open(path, encoding='utf-8') as fp:
                    geojson = json.load(fp)
                    features = geojson['features']
                    map_data = []
                    for ii, feature in enumerate(features):
                        poly = feature['geometry']['coordinates']
                        type = feature['geometry']['type']
                        name = feature['properties']['name']
                        fcode = feature['properties']['FCode']
                        if type == 'MultiLineString':
                            for pp in poly:
                                if len(pp) == 0:
                                    continue
                                data = np.array(pp, dtype=np.float64)
                                if name == '过渡区' or fcode == '99502102':
                                    map_data.append([data, -2])
                                else:
                                    map_data.append([data, -1])
                        elif type == "LineString":
                            if len(poly) == 0:
                                continue
                            data = np.array(poly, dtype=np.float64)
                            if name == '过渡区' or fcode == '99502102':
                                map_data.append([data, -2])
                            else:
                                map_data.append([data, -1])
                        else:
                            print(type)
                    tf_ignore_maps += map_data
            ignore_maps_list.append(tf_ignore_maps)

        for ob_map_paths in self.ob_map_paths_list:
            tf_ob_maps = []
            for ii, path in enumerate(ob_map_paths):
                with open(path, encoding='utf-8') as fp:
                    geojson = json.load(fp)
                    features = geojson['features']
                    map_data = []
                    for ii, feature in enumerate(features):
                        poly = feature['geometry']['coordinates']
                        type = feature['geometry']['type']
                        name = feature['properties']['name']
                        fcode = feature['properties']['FCode']
                        if type == 'MultiLineString':
                            for pp in poly:
                                if len(pp) == 0:
                                    continue
                                data = np.array(pp, dtype=np.float64)
                                map_data.append([data, 1])
                        elif type == "LineString":
                            if len(poly) == 0:
                                continue
                            data = np.array(poly, dtype=np.float64)
                            map_data.append([data, 1])
                        else:
                            print(type)
                    tf_ob_maps += map_data
            ob_maps_list.append(tf_ob_maps)
        return maps_list, ignore_maps_list, ob_maps_list

    def get_scenes(self):
        scenes = []
        # for i, image_paths in enumerate(self.image_paths_list):
        #     image_list = sorted(glob.glob(image_paths))
        #     length = 0
        #     for j, imagename in enumerate(image_list):
        #         data_info = self.data_infos_list[i][imagename.split('/')[-1]]
        #         length += 1
        #         if "end_path" not in data_info:
        #             break
        #     scenes.append([0, length])
        for i, data_infos in enumerate(self.data_infos_list):
            # # image_list = sorted(glob.glob(self.image_paths[i]))
            # length = 0
            # for j, data_info in enumerate(data_infos):
            #     if "end_path" not in data_info:
            #         break
            #     length+= 1
            scenes.append([0, len(data_infos)])
        return scenes

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
            image_list = sorted(glob.glob(image_paths)) #该数据集下所有图像的路径
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
                pose = convert_quat_to_pose_mat(ipose[1:8]) #图像位姿
                vp = vp_infos[imgname]['vp_pose']

                sub_samples.append((image_list[j], pose, vp))
                # sample_list.append([[0, end_idx - self.seq_len + 1], sub_samples, rot, tran])
            sample_list[image_paths] = [sub_samples, rot, tran, K_ori, distorts, H, W, lidar2cam_vector]
        return sample_list
    #结构： {bag_name:[ sub_samples[[image_list[j], pose, vp],], rot, tran, K_ori, distorts, H, W, lidar2cam_vector],}

    def sample_augmentation(self, cam_idx, H, W):
        # H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']

        if self.is_train:
            # if 0:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'][cam_idx])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'][cam_idx])) * newH) - fH
            # crop_h = min(max(0, crop_h), newH - fH-1)
            crop_w = int(np.random.uniform(-fW / 8., fW / 8.) + (newW - fW) / 2.)
            # crop_w = min(max(0, crop_w), newW - fW-1)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # if crop_w + fW > newW or crop_h + fH > newH:
            #    print ('crop = ', crop, newW, newH, fH, fW)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            # print(H, W)
            resize = np.mean(self.data_aug_conf['resize_lim'][cam_idx])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'][cam_idx])) * newH) - fH
            # crop_w = int(max(0, newW - fW) / 2)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            print('crop = ', H, W, crop_w, crop_h, crop, resize, resize_dims)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def ray_box_intersection(self, ray_o, ray_d, near, far, aabb_min=None, aabb_max=None):
        """Returns 1-D intersection point along each ray if a ray-box intersection is detected
        If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary
        Args:
            ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
            ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
            (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
            (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified
        Returns:
            z_ray_in:
            z_ray_out:
            intersection_map: Maps intersection values in z to their ray-box intersection
        """
        # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
        # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
        if aabb_min is None:
            aabb_min = np.ones_like(ray_o) * -1.  # tf.constant([-1., -1., -1.])
        if aabb_max is None:
            aabb_max = np.ones_like(ray_o)  # tf.constant([1., 1., 1.])

        inv_d = np.reciprocal(ray_d)

        t_min = (aabb_min - ray_o) * inv_d
        t_max = (aabb_max - ray_o) * inv_d

        t0 = np.minimum(t_min, t_max)
        t1 = np.maximum(t_min, t_max)

        t_near = np.maximum(np.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
        t_far = np.minimum(np.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

        # Check if rays are inside boxes
        intersection_map = t_far > t_near  # np.where(t_far > t_near)[0]

        # Check that boxes are in front of the ray origin
        positive_far = (t_far * intersection_map) > 0
        # positive_near_far = (t_near * positive_far) > 0
        intersection_map = np.logical_and(intersection_map, positive_far)

        if not intersection_map.shape[0] == 0:
            near[intersection_map, 0] = t_near[intersection_map]  ###光线进入平面处（最靠近的平面）的最大t值
            far[intersection_map, 0] = t_far[intersection_map]  ###光线离开平面处（最远离的平面）的最小t值
        else:
            return None, None, None

        return near, far, intersection_map

    def get_image_data_tf(self, recs, cams, extrin_rot, extrin_tran, K_ori, distorts, H, W):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        aug_params = []
        cam_pos_embeddings = []
        dist_coeffss = []

        intrin = torch.Tensor(K_ori)
        rot = torch.Tensor(extrin_rot)
        tran = torch.Tensor(extrin_tran)
        # distorts = torch.tensor(distorts)

        downsample = 4
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 128,352
        fH, fW = ogfH // downsample, ogfW // downsample
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(1, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(1, fH, fW)
        ds = torch.ones_like(xs)
        rays = torch.stack((xs, ys, ds), -1)

        # Get NewCameraMatrix and map1、map2
        # H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        whole_img_size = (H, W)
        alpha = 1
        # from input to feature after goe_backbone
        K_ori = np.array(K_ori).reshape(3, 3)

        # getOptimalNewCameraMatrix的输入参数为1、原始相机内参，2、畸变参数，3、图像原始尺寸，4、alpha值，5、去畸后的图片尺寸，6、centerPrincipalPoint，
        NewCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(K_ori, distorts, whole_img_size, alpha, whole_img_size)
        # 指定去畸变前后图像尺寸相同，此处可以调整修改，影响不大；
        map1, map2 = cv2.initUndistortRectifyMap(K_ori, distorts, None, NewCameraMatrix, whole_img_size, cv2.CV_32FC1)

        distorts = torch.tensor(distorts)
        # img_ori2 = np.zeros((H*4, W*4, 3))
        # for i in range(W):
        #     for j in range(H):
        #         if i % 1000 == 0 and j % 1000 == 0:
        #             print('xxxxxxx = ', i, j, map2[i,j], map1[i,j])
        #             cv2.line(img_ori2, (int(i*4+2), int(j*4+2)), (int(map2[i,j]*4+2), int(map1[i,j]*4+2)), (255, 255, 0), 10)
        #             cv2.circle(img_ori2, (int(i*4+2), int(j*4+2)), 20, (255, 0, 0), 10)
        #             cv2.circle(img_ori2, (int(map2[i,j]*4+2), int(map1[i,j]*4+2)), 20, (0, 0, 255), 10)
        # cv2.imwrite('./img_ori2.jpg', img_ori2)
        # map1 = self.get_map(H, W, cams, map1, downsample)
        # map2 = self.get_map(H, W, cams, map2, downsample)
        # NewCameraMatrix = torch.tensor(NewCameraMatrix).expand(intrins.shape)
        # xdm
        # print('before_shuff: ', cams)
        # np.random.shuffle(cams)
        # print('after_shuff: ', cams)
        # idxs = random.sample(range(0,2),2)
        idxs = [0]
        np.random.shuffle(idxs)
        for jj in idxs:
            aug_params.append(self.sample_augmentation(jj, H, W))
        for ii, sample in enumerate(recs):

            image_path, _, vp = sample

            image = cv2.imread(image_path)
            # print('image_path = ', image_path, image.shape)  
            for jj in range(len(idxs)):
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                # print('map1.shapexxxxx = ', map1.shape)
                # xx,yy = map1.shape
                # augmentation (resize, crop, horizontal flip, rotate)
                resize, resize_dims, crop, flip, rotate = aug_params[jj]
                img, post_rot2, post_tran2, post_map1, post_map2 = cvimg_transform(image, post_rot, post_tran, vp,
                                                                                   resize=resize,
                                                                                   resize_dims=resize_dims,
                                                                                   crop=crop,
                                                                                   flip=flip,
                                                                                   rotate=rotate,
                                                                                   map1=map1,
                                                                                   map2=map2,
                                                                                   downsample=downsample,
                                                                                   flag=1)

                # # cv2.imwrite('img_amba.jpg', img)
                # for convenience, make augmentation matrices 3x3
                # post_tran2 = post_tran
                # post_rot2 = post_rot
                # img = image
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                # N, _ = post_trans.shape
                # undo post-transformation
                # B x N x D x H x W x 3

                points = rays - post_tran.view(1, 1, 1, 3)
                points = torch.inverse(post_rot).view(1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
                points = points.squeeze().numpy()

                # undistorted points (2 ROI)
                # for i in range(int(N/2)):
                #     points[2*i] = cv2.remap(points[2*i], map1, map2, cv2.INTER_LINEAR)
                #     points[2*i+1] = cv2.remap(points[2*i+1], map1[1], map2[1], cv2.INTER_LINEAR)
                points1 = cv2.remap(points, post_map1, post_map2, cv2.INTER_LINEAR)

                # img_dst = np.zeros((H*4, W*4, 3))
                # for i in range(32):
                #     for j in range(88):
                #         if i % 20 == 0 and j % 20 == 0:
                #             print('yyyyyyy= ', points1[i,j,0], points1[i,j,1], points[i,j, 0], points[i,j,1])
                #             cv2.line(img_dst, (int(points1[i,j,0]*4+2), int(points1[i,j,1]*4+2)), (int(points[i,j,0]*4+2), int(points[i,j,1]*4+2)), (255, 255, 0), 10)
                #             cv2.circle(img_dst, (int(points1[i,j,0]*4+2), int(points1[i,j,1]*4+2)), 20, (255, 0, 0), 10)
                #             cv2.circle(img_dst, (int(points[i,j,0]*4+2), int(points[i,j,1]*4+2)), 20, (0, 0, 255), 10)
                # cv2.imwrite('./img_dsty.jpg', img_dst)

                # img_dst = np.zeros((H*4, W*4, 3))
                # for i in range(88):
                #     for j in range(32):
                #         if i % 1000 == 0 and j % 1000 == 0:
                #             print('yyyyyyy= ', points1[i,j,0], points1[i,j,1], points[i,j, 0], points[i,j,1])
                #             cv2.line(img_dst, (int(points[i,j,1]*4+2), int(points[i,j,0]*4+2)),  (int(post_map2[i,j]*4+2), int(post_map1[i,j]*4+2)), (255, 255, 0), 10)
                #             cv2.circle(img_dst, (int(points[i,j,1]*4+2), int(points[i,j,0]*4+2)), 20, (255, 0, 0), 10)
                #             cv2.circle(img_dst,  (int(post_map2[i,j]*4+2), int(post_map1[i,j]*4+2)), 20, (0, 0, 255), 10)
                # cv2.imwrite('./img_dst_noR.jpg', img_dst)

                # transform to Camera system
                points = torch.tensor(points1).unsqueeze(1).unsqueeze(-1)
                points = torch.inverse(torch.tensor(NewCameraMatrix, dtype=torch.float)).view(1, 1, 1, 3, 3).matmul(
                    points).squeeze(-1)
                cam_pos_embedding = self.positional_encoding(points[..., :2], 4).view(1, fH, fW, -1).permute(0, 3, 1, 2)

                # print ("trans222222:: post_rot = ", post_rot)
                imgs.append(normalize_img(img))
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)
                cam_pos_embeddings.append(cam_pos_embedding)
                dist_coeffss.append(distorts)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(dist_coeffss), torch.stack(post_rots), torch.stack(post_trans),
                torch.stack(cam_pos_embeddings), aug_params)

    def get_localmap_seg(self, sce_id, recs, extrin_rot, extrin_tran):
        # print('sce_id = ', sce_id)
        local_maps = []
        data_infos = self.data_infos_list[sce_id]
        T_lidar2cam = self.T_lidar2cam_list[sce_id]
        roadmap_data = self.roadmap_data_list[sce_id]
        mesh_objs = self.mesh_objs_list[sce_id]
        ignore_maps = self.ignore_maps_list[sce_id]
        ob_maps = []
        if len(self.ob_maps_list) > 0:
            ob_maps = self.ob_maps_list[sce_id]
        maps = self.maps_list[sce_id]
        valid_lf_areas = []
        for ii, sample in enumerate(recs):
            image_path, pose, _ = sample
            # print('imagpath = ', image_path)
            data_info = data_infos[os.path.split(image_path)[-1]]
            concerned_obj_idxs = data_info['concerned_obj_idxs']
            concerned_map_idxs = data_info['concerned_map_idxs']
            concerned_roadmap_idxs = data_info['concerned_roadmap_idxs']
            concerned_ignore_map_idxs = []
            isExist = data_info.get('concerned_ignore_map_idxs', 'not exist')
            if isExist != 'not exist':
                concerned_ignore_map_idxs = data_info['concerned_ignore_map_idxs']  # guodu & ignore
            # concerned_ignore_map_idxs = data_info['concerned_ignore_map_idxs']#guodu & ignore
            concerned_ob_map_idxs = []
            isExist = data_info.get('concerned_ob_map_idxs', 'not exist')
            if isExist != 'not exist':
                concerned_ob_map_idxs = data_info['concerned_ob_map_idxs']  # zhangaiwu
            local_map = [np.zeros((self.nx[0], self.nx[1])) for jj in range(6)]
            # show_seg_map = np.zeros((self.nx[0], self.nx[1], 3), dtype=np.uint8)*255
            # print('nx,ny = ', self.nx[0], self.nx[1])
            for map_idx in concerned_map_idxs:
                poly, label = maps[map_idx]
                # print ('label = ', label)
                if 1:
                    poly = np.concatenate([poly, np.ones_like(poly[:, :1])], axis=-1)
                    poly2lidar = pose.getI().dot(poly.T)
                    poly2cam = T_lidar2cam.dot(poly2lidar)[:3, :]
                    poly2car = extrin_rot.dot(poly2cam).T + extrin_tran
                    pts = poly2car[:, :2]
                    # print ('pts = ', pts,  self.bx[:2], self.dx[:2])
                    pts = np.round(
                        (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                    ).astype(np.int32)
                    # print ('pts00000 = ', pts)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    # print ('pts11111 = ', pts)
                    if label == 5:
                        # cv2.polylines(local_map[label-1], [pts], 0, 1, 2)
                        # cv2.polylines(show_seg_map, [pts], 0, (0, 0, 255), 2)
                        cv2.polylines(local_map[label - 1], [pts], 0, 1, 2)
                    elif label > 0:
                        cv2.fillPoly(local_map[label - 1], [pts], 1)
                        # cv2.fillPoly(show_seg_map, [pts], (0, 255, 255))
                    elif label == -1:  # ignore
                        for jj in range(5):
                            cv2.polylines(local_map[jj], [pts], 1, -1, 10)
                            # cv2.fillPoly(show_seg_map, [pts], (0, 255, 0))
                            # cv2.polylines(local_map[jj], [pts], 1, -1, 1)
                            cv2.fillPoly(local_map[jj], [pts], -1)
            # cl_map = np.zeros((self.nx[0], self.nx[1], 3), dtype=np.uint8)                
            for roadmap_idx in concerned_roadmap_idxs:
                roadmap_idx = int(roadmap_idx)
                # print(self.roadmap_data,  roadmap_idx)
                road = roadmap_data[roadmap_idx]
                if 'cl' in road:
                    poly = road['cl']
                    poly = np.concatenate([poly, np.ones_like(poly[:, :1])], axis=-1)
                    poly2lidar = pose.getI().dot(poly.T)
                    poly2cam = T_lidar2cam.dot(poly2lidar)[:3, :]
                    poly2car = extrin_rot.dot(poly2cam).T + extrin_tran
                    pts = poly2car[:, :2]
                    pts = np.round(
                        (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                    ).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    # print ('pts = ', pts)
                    # cv2.polylines(local_map[5], [pts], 0, 1, 2)
                    # cv2.polylines(show_seg_map, [pts], 0, (255, 255, 255), 1)
                    cv2.polylines(local_map[5], [pts], 0, 1, 1)
            # cv2.imwrite('show_seg_map.jpg', show_seg_map)
            # mesh
            valid_local_area = np.zeros((self.nx[0], self.nx[1]))
            valid_lf_area = np.zeros((self.nx[0], self.nx[1]))
            for obj_idx in concerned_obj_idxs:
                #    if obj_idx >= len(mesh_objs):
                #    print  (len(mesh_objs), obj_idx, image_path)
                mesh_obj = mesh_objs[obj_idx]
                vertices = np.asarray(mesh_obj.vertices)
                triangles = np.asarray(mesh_obj.triangles)
                if triangles.shape[0] == 0:
                    continue

                vertices = np.concatenate([vertices, np.ones_like(vertices[:, :1])], axis=-1)
                vertices2lidar = pose.getI().dot(vertices.T)
                vertices2cam = T_lidar2cam.dot(vertices2lidar)[:3, :]
                vertices2car = extrin_rot.dot(vertices2cam).T + extrin_tran
                pts = vertices2car[:, :2]
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                triangles = list(pts[triangles])
                # for kk in range(triangles.shape[0]):
                #    triangle = pts[triangles[kk]]
                cv2.fillPoly(valid_local_area, triangles, 1)
                cv2.fillPoly(valid_lf_area, triangles, 1)

            # ob
            for map_idx in concerned_ob_map_idxs:
                poly, label = ob_maps[map_idx]
                if 1:
                    if not np.all(poly[0] == poly[-1]):
                        continue
                    poly1 = Polygon(poly)
                    if not poly1.is_valid:
                        continue
                    poly = np.concatenate([poly, np.ones_like(poly[:, :1])], axis=-1)
                    poly2lidar = pose.getI().dot(poly.T)
                    poly2cam = T_lidar2cam.dot(poly2lidar)[:3, :]
                    poly2car = extrin_rot.dot(poly2cam).T + extrin_tran
                    pts = poly2car[:, :2]
                    pts = np.round(
                        (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                    ).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    if label == 1:
                        cv2.fillPoly(valid_local_area, [pts], 1)
                        cv2.fillPoly(valid_lf_area, [pts], 2)

            for map_idx in concerned_ignore_map_idxs:
                poly, label = ignore_maps[map_idx]
                if 1:
                    poly = np.concatenate([poly, np.ones_like(poly[:, :1])], axis=-1)
                    poly2lidar = pose.getI().dot(poly.T)
                    poly2cam = T_lidar2cam.dot(poly2lidar)[:3, :]
                    poly2car = extrin_rot.dot(poly2cam).T + extrin_tran
                    pts = poly2car[:, :2]
                    pts = np.round((pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    # if label == 1:
                    #     cv2.fillPoly(valid_local_area, [pts], 1)
                    #     cv2.fillPoly(valid_lf_area, [pts], 1)
                    if label == -2:
                        cv2.polylines(valid_lf_area, [pts], 1, 0, 10)
                        cv2.fillPoly(valid_lf_area, [pts], 0)
                    if label == -1:
                        cv2.polylines(valid_local_area, [pts], 1, 0, 10)
                        cv2.fillPoly(valid_local_area, [pts], 0)

                        cv2.polylines(valid_lf_area, [pts], 1, 0, 10)
                        cv2.fillPoly(valid_lf_area, [pts], 0)

            # cv2.imwrite('valid_local.jpg', valid_local_area)
            # cv2.imwrite('valid_lf.jpg', valid_lf_area)
            local_map[5][valid_lf_area == 2] = 0
            local_map[5][valid_lf_area == 0] = -1
            for jj in range(5):
                local_map[jj][valid_local_area == 0] = -1

            local_maps.append(torch.Tensor(np.stack(local_map)))
            valid_lf_areas.append(valid_lf_area)
        return torch.stack(local_maps, axis=0), valid_lf_areas

    def generate_lf(self, sce_id, rec, extrin_rot, extrin_tran):
        data_infos = self.data_infos_list[sce_id]
        roadmap_samples = self.roadmap_samples_list[sce_id]
        T_lidar2cam = self.T_lidar2cam_list[sce_id]

        lf_label = np.zeros((1, self.nx[0], self.nx[1])) - 1
        lf_norm = np.zeros((2, 2, self.nx[0], self.nx[1])) - 999
        lf_kappa = np.zeros((2, self.nx[0], self.nx[1]))
        image_path, pose, _ = rec
        data_info = data_infos[os.path.split(image_path)[-1]]
        concerned_roadmap_idxs = data_info['concerned_roadmap_idxs']
        ipose = pose.getI()
        sample_pts = []
        sample_idxs = []
        for roadmap_idx in concerned_roadmap_idxs:
            roadmap_idx = int(roadmap_idx)
            if roadmap_idx not in roadmap_samples:
                continue
            if "left" in roadmap_samples[roadmap_idx]:
                # mesh = np.load(roadmap_samples[roadmap_idx]["left"])
                with open(roadmap_samples[roadmap_idx]["left"], 'rb') as f:
                    mesh = np.load(f, allow_pickle=True)
                ys = np.arange(mesh.shape[0])
                np.random.shuffle(ys)
                ys = ys[:3]
                mesh = mesh[ys]
                xs, ys = np.meshgrid(np.arange(mesh.shape[1]), ys)
                sample_pts.append(np.reshape(mesh, (-1, 6)))
                xs = np.reshape(xs, (-1, 1))
                ys = np.reshape(ys, (-1, 1))
                idxs = np.concatenate(
                    [np.zeros_like(ys, dtype=np.int32) + roadmap_idx, np.zeros_like(ys, dtype=np.int32), ys, xs],
                    axis=-1)
                sample_idxs.append(idxs)
            if "right" in roadmap_samples[roadmap_idx]:
                # mesh = np.load(roadmap_samples[roadmap_idx]["right"])
                with open(roadmap_samples[roadmap_idx]["right"], 'rb') as f:
                    mesh = np.load(f, allow_pickle=True)
                ys = np.arange(mesh.shape[0])
                np.random.shuffle(ys)
                ys = ys[:3]
                mesh = mesh[ys]
                xs, ys = np.meshgrid(np.arange(mesh.shape[1]), ys)
                sample_pts.append(np.reshape(mesh, (-1, 6)))
                xs = np.reshape(xs, (-1, 1))
                ys = np.reshape(ys, (-1, 1))
                idxs = np.concatenate(
                    [np.zeros_like(ys, dtype=np.int32) + roadmap_idx, np.ones_like(ys, dtype=np.int32), ys, xs],
                    axis=-1)
                sample_idxs.append(idxs)
        data2car = None
        idxs = None
        if len(sample_pts) > 0:
            sample_pts = np.concatenate(sample_pts, axis=0)
            idxs = np.concatenate(sample_idxs, axis=0)

            data = sample_pts[..., :3]
            data = np.concatenate([data, np.ones_like(data[:, :1])], axis=-1)
            kappa = np.reshape(sample_pts[..., 3], (-1, 1))
            norm = np.reshape(sample_pts[..., 4:], (-1, 2))
            norm = np.concatenate([norm, np.zeros_like(kappa)], axis=-1)

            data2lidar = ipose.dot(data.T)
            data2cam = T_lidar2cam.dot(data2lidar)[:3, :]
            data2car = (extrin_rot.dot(data2cam).T + extrin_tran).A
            pts = data2car[..., :2]
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            # print(key, data.shape, pts.shape)
            pts[:, [1, 0]] = pts[:, [0, 1]]

            norm2lidar = ipose[:3, :3].dot(norm.T)
            norm2cam = T_lidar2cam[:3, :3].dot(norm2lidar)[:3, :]
            norm2car = extrin_rot.dot(norm2cam).T.A
            norms = norm2car[..., :2]
            norms = norms / np.linalg.norm(norms, axis=-1).reshape((-1, 1))
            norms[:, [1, 0]] = norms[:, [0, 1]]
            norms[:, 1] -= 1

            mask = np.logical_and(np.logical_and(pts[:, 0] >= 0, pts[:, 0] < self.nx[1]),
                                  np.logical_and(pts[:, 1] >= 0, pts[:, 1] < self.nx[0]))
            xs = pts[:, 0][mask]
            ys = pts[:, 1][mask]
            if xs.shape[0] > 0:
                norms = norms[mask]
                kappas = kappa[mask]
                lf_label[:, ys, xs] = 0
                lf_norm[0, :, ys, xs] = norms
                lf_norm[1, :, ys, xs] = norms
                lf_kappa[0, ys, xs] = kappas.flatten()
                lf_kappa[1, ys, xs] = kappas.flatten()
        return lf_label, lf_norm, lf_kappa, data2car, idxs

    def get_localmap_lf(self, sce_id, recs, extrin_rot, extrin_tran, valid_lf_areas):
        # angles = np.linspace(-np.pi/4+np.pi,np.pi/4+np.pi,8).tolist()
        # vectors = np.array([(math.cos(q), math.sin(q)) for q in angles])
        bb = box(self.bx[0] - self.dx[0] / 2., self.bx[1] - self.dx[1] / 2.,
                 self.bx[0] - self.dx[0] / 2. + self.nx[0] * self.dx[0],
                 self.bx[1] - self.dx[1] / 2. + self.nx[1] * self.dx[1])
        fork_labels = []
        fork_patchs = []
        fork_scales = []
        fork_offsets = []
        fork_oris = []

        lf_labels = []
        lf_norms = []
        lf_kappas = []

        data_infos = self.data_infos_list[sce_id]
        T_lidar2cam = self.T_lidar2cam_list[sce_id]
        roadmap_forks = self.roadmap_forks_list[sce_id]
        for ii, sample in enumerate(recs):
            # print ('ii = ', ii)
            valid_lf_area = valid_lf_areas[ii]
            image_path, pose, _ = sample
            data_info = data_infos[os.path.split(image_path)[-1]]
            # concerned_roadmap_idxs = data_info['concerned_roadmap_idxs']
            ipose = pose.getI()
            lf_label, lf_norm, lf_kappa, lf_sample_pts, lf_sample_idxs = self.generate_lf(sce_id, sample, extrin_rot,
                                                                                          extrin_tran)
            fork_label = np.zeros((self.nx[0], self.nx[1]))
            fork_patch = np.zeros((2, self.crop_size, self.crop_size)) - 1.
            fork_scale = np.ones((1))
            fork_offset = np.zeros((2))
            fork_ori = np.zeros((1))
            fork_sample_idxs = []
            fork_sample_pts = []
            concerned_roadmap_forks = data_info['concerned_roadmap_forks']

            for key in concerned_roadmap_forks:
                area = np.array(roadmap_forks[key]['area'], dtype=np.float64)
                area = np.concatenate([area, np.ones_like(area[:, :1])], axis=-1)
                area2lidar = ipose.dot(area.T)
                area2cam = T_lidar2cam.dot(area2lidar)[:3, :]
                area2car = extrin_rot.dot(area2cam).T + extrin_tran
                area2car = area2car[:, :2]
                area = Polygon(area2car)
                if bb.intersects(area):
                    pts = np.round(
                        (area2car - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                    ).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    cv2.fillPoly(fork_label, [pts], 1)
                    fork_sample_idxs.append(roadmap_forks[key]['idxs'])
                    fork_sample_pts.append(roadmap_forks[key]['pts'])

            # !!!!!
            lf_label[:, valid_lf_area == 1] = 0
            lf_label[:, fork_label == 1] = -1

            if len(fork_sample_idxs) > 0:
                fork_sample_idxs = np.concatenate(fork_sample_idxs)
                fork_sample_pts = np.concatenate(fork_sample_pts)
                # print('fork_sample_pts = ', fork_sample_pts, fork_sample_pts.shape)
                fork_sample_kappa0 = fork_sample_pts[..., 3]
                fork_sample_norm0 = np.reshape(fork_sample_pts[..., 4:6], (-1, 2))
                fork_sample_norm0 = np.concatenate([fork_sample_norm0, np.zeros_like(fork_sample_norm0[..., 0:1])],
                                                   axis=-1)

                fork_sample_kappa1 = fork_sample_pts[..., 6]
                fork_sample_norm1 = np.reshape(fork_sample_pts[..., 7:9], (-1, 2))
                fork_sample_norm1 = np.concatenate([fork_sample_norm1, np.zeros_like(fork_sample_norm1[..., 0:1])],
                                                   axis=-1)

                fork_sample_pts = np.concatenate([fork_sample_pts[..., :3], np.ones_like(fork_sample_pts[:, :1])],
                                                 axis=-1)

                fork_sample_pts2lidar = ipose.dot(fork_sample_pts.T)
                fork_sample_pts2cam = T_lidar2cam.dot(fork_sample_pts2lidar)[:3, :]
                fork_sample_pts2car = (extrin_rot.dot(fork_sample_pts2cam).T + extrin_tran).A

                fork_pts = fork_sample_pts2car[..., :2]
                fork_pts = np.round(
                    (fork_pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                ).astype(np.int32)
                # print(key, data.shape, pts.shape)
                fork_pts[:, [1, 0]] = fork_pts[:, [0, 1]]

                fork_sample_norm0_2lidar = ipose[:3, :3].dot(fork_sample_norm0.T)
                fork_sample_norm0_2cam = T_lidar2cam[:3, :3].dot(fork_sample_norm0_2lidar)[:3, :]
                fork_sample_norm0_2car = extrin_rot.dot(fork_sample_norm0_2cam).T.A
                fork_sample_norm0_2car = fork_sample_norm0_2car[..., :2]
                fork_sample_norm0_2car = fork_sample_norm0_2car / np.linalg.norm(fork_sample_norm0_2car,
                                                                                 axis=-1).reshape((-1, 1))
                fork_sample_norm0_2car[:, [1, 0]] = fork_sample_norm0_2car[:, [0, 1]]

                fork_sample_norm1_2lidar = ipose[:3, :3].dot(fork_sample_norm1.T)
                fork_sample_norm1_2cam = T_lidar2cam[:3, :3].dot(fork_sample_norm1_2lidar)[:3, :]
                fork_sample_norm1_2car = extrin_rot.dot(fork_sample_norm1_2cam).T.A
                fork_sample_norm1_2car = fork_sample_norm1_2car[..., :2]
                fork_sample_norm1_2car = fork_sample_norm1_2car / np.linalg.norm(fork_sample_norm1_2car,
                                                                                 axis=-1).reshape((-1, 1))
                fork_sample_norm1_2car[:, [1, 0]] = fork_sample_norm1_2car[:, [0, 1]]

                mask = np.logical_and(np.logical_and(fork_pts[:, 0] >= 0, fork_pts[:, 0] < self.nx[1]),
                                      np.logical_and(fork_pts[:, 1] >= 0, fork_pts[:, 1] < self.nx[0]))
                fork_sample_norm0_2car = fork_sample_norm0_2car[mask]
                fork_sample_norm1_2car = fork_sample_norm1_2car[mask]
                fork_sample_kappa0 = fork_sample_kappa0[mask]
                fork_sample_kappa1 = fork_sample_kappa1[mask]

                fork_pts = fork_pts[mask]
                tmp = np.cross(fork_sample_norm0_2car, fork_sample_norm1_2car)
                ys = fork_pts[..., 1]
                xs = fork_pts[..., 0]
                fork_sample_norm0_2car[:, 1] -= 1
                fork_sample_norm1_2car[:, 1] -= 1
                lf_label[:, ys, xs] = 1
                lf_norm[(tmp > 0).astype(np.int32), :, ys, xs] = fork_sample_norm0_2car
                lf_norm[(tmp <= 0).astype(np.int32), :, ys, xs] = fork_sample_norm1_2car
                lf_kappa[(tmp > 0).astype(np.int32), ys, xs] = fork_sample_kappa0
                lf_kappa[(tmp <= 0).astype(np.int32), ys, xs] = fork_sample_kappa1

            #     mask = np.logical_and(np.logical_and(fork_sample_pts2car[:, 0] >=10., fork_sample_pts2car[:, 0] <110.), np.logical_and(fork_sample_pts2car[:, 1] >=-10., fork_sample_pts2car[:, 1] <10.))
            #     idxs = np.where(mask)[0]
            #     if idxs.shape[0] > 0:
            #         fork_patch = fork_patch*0.
            #         idx = idxs[random.randint(0, idxs.shape[0]-1)]
            #         sample_idx = fork_sample_idxs[idx]
            #         sample_pt2car = fork_sample_pts2car[idx][:2].reshape((1, -1))

            #         fork_offset = sample_pt2car[0]

            #         poly0 = np.load(self.roadmap_samples[sample_idx[0]]["left" if sample_idx[1]==0 else "right"])[sample_idx[2]][..., :3]
            #         poly1 = np.load(self.roadmap_samples[sample_idx[3]]["left" if sample_idx[4]==0 else "right"])[sample_idx[5]][..., :3]

            #         poly0 = np.concatenate([poly0, np.ones_like(poly0[:, :1])], axis=-1)
            #         poly0_2lidar = ipose.dot(poly0.T)
            #         poly0_2cam = self.T_lidar2cam.dot(poly0_2lidar)[:3, :]
            #         poly0_2car = extrin_rot.dot(poly0_2cam).T + extrin_tran
            #         poly0_2offset = poly0_2car[:, :2] - sample_pt2car
            #         pts0 = np.round(
            #                     poly0_2offset/fork_scale/ self.dx[:2] + np.array([[self.crop_size/2., self.crop_size/2.]], dtype=np.float64)
            #                     ).astype(np.int32)
            #         pts0[:, [1, 0]] = pts0[:, [0, 1]]

            #         poly1 = np.concatenate([poly1, np.ones_like(poly1[:, :1])], axis=-1)
            #         poly1_2lidar = ipose.dot(poly1.T)
            #         poly1_2cam = self.T_lidar2cam.dot(poly1_2lidar)[:3, :]
            #         poly1_2car = extrin_rot.dot(poly1_2cam).T + extrin_tran
            #         poly1_2offset = poly1_2car[:, :2] - sample_pt2car
            #         pts1 = np.round(
            #                     poly1_2offset / self.dx[:2] +  np.array([[48., 48.]], dtype=np.float64)
            #                     ).astype(np.int32)
            #         pts1[:, [1, 0]] = pts1[:, [0, 1]]

            #         cross = np.cross(poly0_2offset[-1]-poly0_2offset[0], poly1_2offset[-1]-poly1_2offset[0])
            #         if cross > 0:
            #             cv2.polylines(fork_patch[0], [pts0], 0, 1, 1)
            #             cv2.polylines(fork_patch[1], [pts1], 0, 1, 1)
            #         else:
            #             cv2.polylines(fork_patch[1], [pts0], 0, 1, 1)
            #             cv2.polylines(fork_patch[0], [pts1], 0, 1, 1)

            #         min_y0 = pts0[:, 1].min()
            #         max_y0 = pts0[:, 1].max()
            #         min_y1 = pts1[:, 1].min()
            #         max_y1 = pts1[:, 1].max()
            #         min_y = max(min_y0, min_y1)
            #         max_y = min(max_y0, max_y1)
            #         if min_y > 0:
            #             fork_patch[:, :min_y] = -1
            #         if max_y < self.crop_size:
            #             fork_patch[:, max_y:] = -1
            #     # else:
            #     #     print("fork outside!!")

            # elif not lf_sample_pts is None:
            #     mask = np.logical_and(np.logical_and(lf_sample_pts[:, 0] >=10., lf_sample_pts[:, 0] <110.), np.logical_and(lf_sample_pts[:, 1] >=-10., lf_sample_pts[:, 1] <10.))
            #     idxs = np.where(mask)[0]
            #     if idxs.shape[0] > 0:
            #         idx = idxs[random.randint(0, idxs.shape[0]-1)]
            #         sample_idx = lf_sample_idxs[idx]
            #         sample_pt = lf_sample_pts[idx]
            #         mask = np.logical_and(np.logical_and(lf_sample_idxs[..., 0] == sample_idx[0], lf_sample_idxs[..., 1] == sample_idx[1]), lf_sample_idxs[..., 2] == sample_idx[2])
            #         #print(self.roadmap_samples[sample_idx[0]]["left"].shape, self.roadmap_samples[sample_idx[0]]["right"].shape, np.sum(mask), sample_idx, np.sum(mask1))
            #         poly2car = lf_sample_pts[mask]

            #         if poly2car.shape[0] > 0:
            #             fork_patch = fork_patch*0.
            #             sample_pt2car = sample_pt[:2].reshape((1, -1))
            #             fork_offset = sample_pt2car[0]
            #             pts = poly2car[:, :2] - sample_pt2car
            #             pts = np.round(
            #                         pts/fork_scale/ self.dx[:2] + np.array([[self.crop_size/2., self.crop_size/2.]], dtype=np.float64)
            #                         ).astype(np.int32)

            #             pts[:, [1, 0]] = pts[:, [0, 1]]

            #             cv2.polylines(fork_patch[0], [pts], 0, 1, 1)
            #             cv2.polylines(fork_patch[1], [pts], 0, 1, 1)
            #             min_y = pts[:, 1].min()
            #             max_y = pts[:, 1].max()
            #             if min_y > 0:
            #                 fork_patch[:, :min_y] = -1
            #             if max_y < self.crop_size:
            #                 fork_patch[:, max_y:] = -1
            #         else:
            #             print("road outside error!!")
            #     # else:
            #     #     print("road outside!!")
            lf_label[:, valid_lf_area == 0] = -1
            lf_label[:, valid_lf_area == 2] = 0

            lf_norm[0, :, valid_lf_area == 0] = -999
            lf_norm[1, :, valid_lf_area == 0] = -999

            lf_norm[0, :, valid_lf_area == 2] = -999
            lf_norm[1, :, valid_lf_area == 2] = -999

            lf_labels.append(torch.Tensor(lf_label))
            lf_norms.append(torch.Tensor(lf_norm).view(-1, self.nx[0], self.nx[1]))
            lf_kappas.append(torch.Tensor(lf_kappa))

            fork_patchs.append(torch.Tensor(fork_patch))
            fork_scales.append(torch.Tensor(fork_scale))
            fork_offsets.append(torch.Tensor(fork_offset))
            fork_oris.append(torch.Tensor(fork_ori))
        return torch.stack(lf_labels, axis=0), torch.stack(lf_norms, axis=0), torch.stack(lf_kappas,
                                                                                          axis=0), torch.stack(
            fork_patchs, axis=0), torch.stack(fork_scales, axis=0), torch.stack(fork_offsets, axis=0), torch.stack(
            fork_oris, axis=0)

    # def get_map(self, H, W, cams, map, downsample):
    #     maps = []
    #     for jj, cam in enumerate(cams):
    #         resize, resize_dims, crop, flip, rotate = self.sample_augmentation(jj, H, W)
    #         map = map_transform(map, resize_dims, crop, rotate, downsample, flag = 1)
    #         maps.append(torch.tensor(map))
    #     return np.stack(maps)

    def positional_encoding(self, input, L):  # [B,...,N]
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32) * np.pi  # [L]
        spectrum = input[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        return input_enc

    # def get_cam_pos_embedding(self, recs, cams, intrins, distorts, post_trans, post_rots, L):
    #     ogfH, ogfW = self.data_aug_conf['final_dim']
    #     fH, fW = ogfH // self.downsample, ogfW // self.downsample
    #     xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(1, fH, fW)
    #     ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(1, fH, fW)
    #     ds = torch.ones_like(xs)
    #     rays = torch.stack((xs, ys, ds), -1)

    #     # Get NewCameraMatrix and map1、map2
    #     H, W, _ = cv2.imread(recs[0]).shape
    #     whole_img_size = (W, H)
    #     alpha    = 1
    #     downsample = 4 # from input to feature after goe_backbone
    #     intrin  = intrins.numpy()[0] # seqs have the same intrins, get first;
    #     distort = distorts.numpy()[0] # seqs have the same distorts, get first;
    #     #getOptimalNewCameraMatrix的输入参数为1、原始相机内参，2、畸变参数，3、图像原始尺寸，4、alpha值，5、去畸后的图片尺寸，6、centerPrincipalPoint，
    #     NewCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(intrin, distort, whole_img_size, alpha, whole_img_size)
    #     # 指定去畸变前后图像尺寸相同，此处可以调整修改，影响不大；
    #     map1, map2 = cv2.initUndistortRectifyMap(intrin, distort, None, NewCameraMatrix, whole_img_size, cv2.CV_32FC1)
    #     map1 = self.get_map(H, W, cams, map1, downsample)
    #     map2 = self.get_map(H, W, cams, map2, downsample)
    #     NewCameraMatrix = torch.tensor(NewCameraMatrix).expand(intrins.shape)

    #     N, _ = post_trans.shape
    #     # undo post-transformation
    #     # B x N x D x H x W x 3
    #     points = rays - post_trans.view(N, 1, 1, 1, 3)
    #     points = torch.inverse(post_rots).view(N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
    #     points = points.squeeze().numpy()

    #     # undistorted points (2 ROI)
    #     for i in range(int(N/2)):
    #         points[2*i] = cv2.remap(points[2*i], map1[0], map2[0], cv2.INTER_LINEAR)
    #         points[2*i+1] = cv2.remap(points[2*i+1], map1[1], map2[1], cv2.INTER_LINEAR)

    #     # transform to Camera system
    #     points = torch.tensor(points).unsqueeze(1).unsqueeze(-1)
    #     points = torch.inverse(NewCameraMatrix).view(N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    #     cam_pos_embedding = self.positional_encoding(points[..., :2], L).view(N, fH, fW, -1).permute(0, 3, 1, 2)
    #     # 最好再把原始的x,y也保留并concat起来
    #     return cam_pos_embedding

    def get_rays_data_bak(self, sce_id, recs, extrin_rot, extrin_tran, K_ori, distorts, H, W, aug_params):
        rays_all = []
        aabb_min = self.bx2 - self.dx2 / 2
        aabb_max = self.bx2 - self.dx2 / 2 + self.dx2 * (self.nx2 - 1)
        T_lidar2cam = self.T_lidar2cam_list[sce_id]
        for ii, sample in enumerate(recs):
            image_path, _, vp = sample  # pose: world to lidar
            img_name = 1.
            image = cv2.imread(image_path)
            depth_file = image_path.replace("org/img_ori", "depths_single_with_obj_time_ori").replace("/share/",
                                                                                                      "/share/occ_data/occ_data_ori/").replace(
                ".jpg", ".npy")
            with open(depth_file, 'rb') as f:
                depth_org = np.load(f, allow_pickle=True)[:, :3]
            if 0:
                for ii in range(depth_org.shape[0]):
                    ptx = round(depth_org[ii, 0])
                    pty = round(depth_org[ii, 1])
                    cv2.circle(image, (ptx, pty), 3, (0, 0, 0), -1)
                # cv2.imshow('image', image)
                # cv2.waitKey(100)
                cv2.imwrite('1.jpg', image)

            img_for_rays = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) / 255
            depth_pre = depth_org
            intrin = torch.Tensor(K_ori)
            rot = torch.Tensor(extrin_rot)
            tran = torch.Tensor(extrin_tran)
            cam2car = torch.cat([rot, tran.reshape(-1, 1)], 1)
            cam2car[:, 1:3] *= -1
            min_idx = np.argmin(np.array(aug_params, dtype=object)[..., 0])
            resize, resize_dims, crop, flip, rotate = aug_params[min_idx]
            effect_crop = (np.array(crop) / resize).astype(np.int32)
            effect_crop = [max(0, effect_crop[0]), max(0, effect_crop[1]), min(W, effect_crop[2]),
                           min(H, effect_crop[3])]
            kept_depth = (depth_pre[:, 0] < effect_crop[2] - 1) * (depth_pre[:, 0] > effect_crop[0]) * (
                    depth_pre[:, 1] < effect_crop[3] - 1) * (depth_pre[:, 1] > effect_crop[1])
            # depth_pre = depth_pre[np.where(kept_depth)]
            ### filter lidar points outside bev bbox
            ### depth_pre from pixel coordinate system
            depth_temp = copy.deepcopy(depth_org)

            # depth_temp[:, :2] = depth_temp[:, :2] * np.repeat(depth_temp[:, 2:3], 2, 1)
            # intrinsic_paras_i = np.matrix(np.array(K_ori).reshape((3, 3))).I
            # depths_2_cam = np.dot(intrinsic_paras_i, depth_temp.T)

            depth_temp[:, :2] = cv2.undistortPointsIter(depth_temp[:, :2].reshape(-1, 1, 2), np.array(K_ori), distorts,
                                                        None, None, (
                                                            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40,
                                                            0.03)).reshape(-1, 2)
            depth_temp[:, :2] = depth_temp[:, :2] * np.repeat(depth_org[:, 2:3], 2, 1)
            depths_2_cam = depth_temp.T

            ### depth_pre from camera coordinate system
            depths_2_car = np.array(rot.numpy().dot(depths_2_cam).T + tran.numpy())
            kept_depth2 = (depths_2_car[:, 0] >= self.grid_conf['xbound'][0]) * (
                    depths_2_car[:, 0] <= (self.grid_conf['xbound'][1] - self.grid_conf['xbound'][2])) * (
                                  depths_2_car[:, 1] >= self.grid_conf['ybound'][0]) * (depths_2_car[:, 1] <= (
                    self.grid_conf['ybound'][1] - self.grid_conf['ybound'][2])) * (
                                  depths_2_car[:, 2] >= self.grid_conf['zbound'][0]) * (
                                  depths_2_car[:, 2] <= (self.grid_conf['zbound'][1] - self.grid_conf['zbound'][2]))
            depth_pre = depth_pre[np.where(kept_depth2)]
            ###

            ### debug
            if 0:
                pcd_cam = o3d.geometry.PointCloud()
                pcd_cam.points = o3d.utility.Vector3dVector(np.array(depths_2_cam.T))
                pcd_cam.colors = o3d.utility.Vector3dVector(
                    [[0, 255, 0] for i in range(np.array(depths_2_cam.T).shape[0])])

                pcd_car = o3d.geometry.PointCloud()
                pcd_car.points = o3d.utility.Vector3dVector(depth_org2)
                pcd_car.colors = o3d.utility.Vector3dVector([[0, 0, 255] for i in range(depth_org2.shape[0])])

                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name='pcd_car', width=352 * 3, height=128 * 3)
                vis.get_render_option().background_color = np.asarray([0.23, 0.64, 1])  # 设置一些渲染属性
                vis.add_geometry(pcd_cam, True)
                vis.add_geometry(pcd_car, True)

                view_control = vis.get_view_control()
                view_control.set_front([-1, 0, 0])
                view_control.set_lookat([10, 0, 2])
                view_control.set_up([0, 0, 1])
                view_control.set_zoom(0.025)
                view_control.rotate(0, 2100 / 40)

                vis.update_renderer()
                vis.run()
            ###

            depth_arr = torch.Tensor(depth_pre)
            depths_len = len(depth_arr)
            depths = depth_arr[:, 2].reshape(-1, 1)

            if self.is_train:
                seg_file = image_path.replace("org/img_ori", "semantic_maps_ori").replace("/share/",
                                                                                          "/share/occ_data/occ_data_ori/").replace(
                    ".jpg", ".png")
                if os.path.exists(seg_file):
                    semantic_pre = cv2.imread(seg_file, 0)
                    semantic_map = torch.Tensor(semantic_pre)
                else:
                    semantic_map = torch.zeros(H, W, dtype=torch.int8)

                """方案二: 根据depth的数量选取"""
                valid_crop_area = torch.zeros(H, W, dtype=torch.bool)
                valid_sky_area = torch.zeros(H, W, dtype=torch.bool)
                valid_crop_area[effect_crop[1]:effect_crop[3], effect_crop[0]:effect_crop[2]] = 1
                valid_sky_area[semantic_map == 2] = True
                sky_idxs = torch.nonzero((valid_sky_area * valid_crop_area) == True)

                if sky_idxs.size(0) > 0:
                    if sky_idxs.size(0) > depths_len:
                        sample_sky_idx = torch.randint(0, sky_idxs.size(0), (depths_len, 1))[:, 0]
                        sky_idxs = sky_idxs[sample_sky_idx, :]
                    # coords_x = torch.cat([depth_arr[:, 0].long(), sky_idxs[:, 1].long()], dim=0)
                    # coords_y = torch.cat([depth_arr[:, 1].long(), sky_idxs[:, 0].long()], dim=0)
                    coords_x = torch.cat([depth_arr[:, 0], sky_idxs[:, 1]], dim=0)
                    coords_y = torch.cat([depth_arr[:, 1], sky_idxs[:, 0]], dim=0)
                    depths = torch.cat([depths, torch.zeros(sky_idxs.size(0), 1)])
                else:
                    # coords_x = depth_arr[:, 0].long()
                    # coords_y = depth_arr[:, 1].long()
                    coords_x = depth_arr[:, 0]
                    coords_y = depth_arr[:, 1]

                # coords_x = torch.clamp(coords_x, 0, W - 1)
                # coords_y = torch.clamp(coords_y, 0, H - 1)
                coords_x_i = np.round(coords_x).long()
                coords_y_i = np.round(coords_y).long()
                coords_x_i = torch.clamp(coords_x_i, 0, W - 1)
                coords_y_i = torch.clamp(coords_y_i, 0, H - 1)
                coords = torch.cat([coords_x.reshape(-1, 1), coords_y.reshape(-1, 1)], dim=1)
                # print('max coorx:', torch.max(coords[..., 0]), 'min coorx:', torch.min(coords[..., 0]))
                # print('max coory:', torch.max(coords[..., 1]), 'min coory:', torch.min(coords[..., 1]))
                weights = torch.zeros(depths.shape)
                weights[:depths_len, 0] = 1
                img_for_rays = img_for_rays[coords_y_i, coords_x_i, :]
                rays_mask = semantic_map[coords_y_i, coords_x_i].reshape(-1, 1)
                near, far = 1.0 * torch.ones_like(depths), 120.0 * torch.ones_like(depths)

                # 训练时随机选取crop区域内self.sample_num个点, 0: lidar_ray_o, 1: cam_ray_o
                ### get rays from lidar
                sample_num0 = self.sample_num // 2
                # print('depths:', depths.shape, 'depths_len:', depths_len, 'sample_num0:', sample_num0)
                sample_idx0 = torch.randint(0, depths[:depths_len, :].size(0), (sample_num0, 1))[:, 0]
                coords0 = coords[sample_idx0, :]
                depths0 = depths[sample_idx0, :]
                pts_cam0 = depth_arr[sample_idx0, 2:]
                weights0 = weights[sample_idx0, :]
                img_for_rays0 = img_for_rays[sample_idx0, :]
                rays_mask0 = rays_mask[sample_idx0, :]
                near0, far0 = near[sample_idx0, :], far[sample_idx0, :]
                types0 = torch.zeros(sample_num0, 1)
                # trans_lidar2cam = [-0.000888, -0.668762, -1.201224]
                trans_lidar2cam = np.array(T_lidar2cam[:3, 3]).reshape(-1)
                rays_o_for_depth0, rays_d_for_depth0, depths0 = get_rays_from_lidar(K_ori, trans_lidar2cam, cam2car,
                                                                                    torch.cat([coords0, depths0], 1),
                                                                                    distorts)

                ### get rays from cam
                sample_num1 = self.sample_num - sample_num0
                sample_idx1 = torch.randint(0, depths.size(0), (sample_num1, 1))[:, 0]
                coords1 = coords[sample_idx1, :]
                depths1 = depths[sample_idx1, :]
                weights1 = weights[sample_idx1, :]
                img_for_rays1 = img_for_rays[sample_idx1, :]
                rays_mask1 = rays_mask[sample_idx1, :]
                near1, far1 = near[sample_idx1, :], far[sample_idx1, :]
                types1 = torch.ones(sample_num1, 1)
                directions = get_ray_directions(H, W, K_ori, coords1, self.is_train, distorts)
                rays_o_for_depth1, rays_d_for_depth1 = get_rays(directions, cam2car)
                rays_d_for_depth = directions @ cam2car[:, :3].T
                dir_norm = torch.norm(rays_d_for_depth, dim=-1, keepdim=True)
                depths1 = depths1 * dir_norm

                rays_o = torch.cat([rays_o_for_depth0, rays_o_for_depth1])
                rays_d = torch.cat([rays_d_for_depth0, rays_d_for_depth1])
                near = torch.cat([near0, near1])
                far = torch.cat([far0, far1])
                rays_mask = torch.cat([rays_mask0, rays_mask1])
                depths = torch.cat([depths0, depths1])
                weights = torch.cat([weights0, weights1])
                img_for_rays = torch.cat([img_for_rays0, img_for_rays1])
                types = torch.cat([types0, types1])

                near_ins, far_ins, intersection_map = self.ray_box_intersection(np.array(rays_o), np.array(rays_d),
                                                                                near.numpy().copy(), far.numpy().copy(),
                                                                                aabb_min, aabb_max)
                near_ins, far_ins = torch.Tensor(near_ins).to(depths.device), torch.Tensor(far_ins).to(depths.device)
                near, far = near_ins, far_ins

                img_names = torch.Tensor([[img_name]] * near.size(0))
                rays = torch.cat(
                    [
                        rays_o,
                        rays_d,
                        near,
                        far,
                        rays_mask,
                        depths,
                        weights,
                        img_names,
                        img_for_rays,  # rgb
                        types
                    ],
                    1,
                )  # (h*w, 15)
                valid_index = rays[:, 6] < rays[:, 7]
                rays = rays[valid_index]
                sample_idx = torch.randint(0, rays.size(0), (self.sample_num, 1))[:, 0]
                rays = rays[sample_idx]

                if vis_depth:
                    image_name = image_path.split('/')[-1][:-4]
                    print(f"saving... at samples/depth/{image_name}.ply")
                    os.makedirs("samples/depth", exist_ok=True)
                    pts = rays_o + rays_d * depths
                    gt_pcd = o3d.geometry.PointCloud()
                    gt_pcd.points = o3d.utility.Vector3dVector(pts.numpy())
                    o3d.io.write_point_cloud(
                        f"samples/depth/{image_name}.ply", gt_pcd
                    )
                    # np.save(f"samples/depth/{image_name}_dir_norm.npy", dir_norm.numpy())
            else:
                if 0:
                    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
                    grid = grid[effect_crop[1]:effect_crop[3], effect_crop[0]:effect_crop[2], :]
                    directions = get_ray_directions(H, W, K_ori, grid.reshape(-1, 2), True, distorts)
                    rays_o, rays_d = get_rays(directions, cam2car)

                    directions_for_depth = get_ray_directions(H, W, K_ori, depth_arr[:, :2], True, distorts)
                    rays_o_for_depth, rays_d_for_depth = get_rays(directions_for_depth, cam2car)

                    """generate pcd GT"""""
                    rays_d_for_depth2 = directions_for_depth @ cam2car[:, :3].T
                    dir_norm = torch.norm(rays_d_for_depth2, dim=-1, keepdim=True)
                    depths = depths * dir_norm
                    pts_gt = rays_o_for_depth + rays_d_for_depth * depths
                    # pts_gt = torch.cat([pts_gt,torch.zeros(rays_o.size(0)-depths_len,3)])
                    """generate pcd GT"""""

                    """求解rays与voxel的两个交点"""
                    near, far = 0. * torch.ones_like(directions_for_depth[..., :1]), 120.0 * torch.ones_like(
                        directions_for_depth[..., :1])
                    near_ins, far_ins, _ = self.ray_box_intersection(np.array(rays_o_for_depth),
                                                                     np.array(rays_d_for_depth), near.numpy().copy(),
                                                                     far.numpy().copy(), aabb_min, aabb_max)
                    near_ins, far_ins = torch.Tensor(near_ins).to(depths.device), torch.Tensor(far_ins).to(
                        depths.device)
                    near, far = near_ins, torch.min(depths + 3, far_ins)
                    img_names = torch.cat([torch.Tensor([[img_name]]), torch.zeros(near.size(0) - 1, 1)])
                    rays = torch.cat(
                        [
                            rays_o_for_depth,
                            rays_d_for_depth,
                            near,
                            far,
                            img_names,
                            depths,
                            pts_gt
                        ],
                        1,
                    )  # (h*w, 11)

                if 1:
                    select_num = 25000
                    directions_for_depth = get_ray_directions(H, W, K_ori, depth_arr[:, :2], True, distorts)
                    rays_o_for_depth, rays_d_for_depth = get_rays(directions_for_depth, cam2car)
                    rays_d_for_depth2 = directions_for_depth @ cam2car[:, :3].T
                    dir_norm = torch.norm(rays_d_for_depth2, dim=-1, keepdim=True)
                    depths = depths * dir_norm
                    pts_gt = rays_o_for_depth + rays_d_for_depth * depths
                    rays = torch.zeros((select_num, 13))
                    if pts_gt.size(0) > select_num:
                        rays[:, 10:13] = pts_gt[:select_num, :]
                    else:
                        rays[:pts_gt.size(0), 10:13] = pts_gt[:pts_gt.size(0), :]

                ### debug
                if 0:
                    sample_num = 20000
                    # trans_lidar2cam = [-0.000888, -0.668762, -1.201224]
                    trans_lidar2cam = np.array(T_lidar2cam[:3, 3]).reshape(-1)
                    rays_o_for_depth0, rays_d_for_depth0, depths0 = get_rays_from_lidar(K_ori, trans_lidar2cam, cam2car,
                                                                                        torch.cat([coords0, depths0],
                                                                                                  1), distorts)
                    z_val_lidar = torch.linspace(0.0, 1.0, 100) * depths[0:sample_num]
                    ray_lidar_o = (
                            rays_o_for_depth[0:sample_num].unsqueeze(-2) + rays_d_for_depth[0:sample_num].unsqueeze(
                        -2) * z_val_lidar.unsqueeze(-1)).view(-1, 3)
                    pcd_lidar_o = o3d.geometry.PointCloud()
                    pcd_lidar_o.points = o3d.utility.Vector3dVector(ray_lidar_o.cpu().numpy())
                    pcd_lidar_o.colors = o3d.utility.Vector3dVector([[0, 0, 255] for i in range(ray_lidar_o.shape[0])])

                    z_val_cam = torch.linspace(0.0, 1.0, 100) * depths_cam_o[0:sample_num]
                    ray_cam_o = (rays_o[0:sample_num].unsqueeze(-2) + rays_d[0:sample_num].unsqueeze(
                        -2) * z_val_cam.unsqueeze(-1)).view(-1, 3)
                    pcd_cam_o = o3d.geometry.PointCloud()
                    pcd_cam_o.points = o3d.utility.Vector3dVector(ray_cam_o.cpu().numpy())
                    pcd_cam_o.colors = o3d.utility.Vector3dVector([[0, 255, 0] for i in range(z_val_cam.shape[0])])

                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name='pts_gt', width=352 * 3, height=128 * 3)
                    vis.get_render_option().background_color = np.asarray([0.23, 0.64, 1])  # 设置一些渲染属性
                    vis.add_geometry(pcd_cam_o, True)
                    vis.add_geometry(pcd_lidar_o, True)

                    view_control = vis.get_view_control()
                    view_control.set_front([-1, 0, 0])
                    view_control.set_lookat([10, 0, 2])
                    view_control.set_up([0, 0, 1])
                    view_control.set_zoom(0.025)
                    view_control.rotate(0, 2100 / 40)

                    vis.update_renderer()
                    vis.run()
                ###

                if rays.size(0) < select_num:
                    rays = torch.cat([rays, torch.zeros(select_num - rays.size(0), 13)])
                else:
                    sample_idx = torch.randint(0, rays.size(0), (select_num, 1))[:, 0]
                    rays = rays[sample_idx, :]

            rays_all.append(rays)

        return torch.stack(rays_all)

    def projectPoints(self, proj_points, dist_coeffs, ori_intrin):
        """
        :param data: (torch.tensor, shape=[N, cams, C, 3]) 3D points in camera coordinates.
        :param K: (torch.tensor, shape=[N, cams, 3, 3]) Camera matrix.
        :param dist_coeffs: (torch.tensor, shape=[N, cams, 1, 8]) Distortion coefficients.
        : 径向畸变系数(k1,k2),切向畸变系数(p1,p2),径向畸变系数(k3,k4,k5,k6)
        :return: (torch.tensor, shape=[N, 2]) Projected 2D points.
        """
        # Apply dist_coeffs
        r = torch.sum(proj_points[..., :2] ** 2, dim=-1, keepdim=True).squeeze(-1)
        k1, k2, p1, p2, k3, k4, k5, k6 = dist_coeffs.unbind(-1)
        radial = (1.0 + k1 * r + k2 * r ** 2 + k3 * r ** 3) / (1 + k4 * r + k5 * r ** 2 + k6 * r ** 3)
        tangential1 = 2.0 * p1 * proj_points[..., 0] * proj_points[..., 1] + p2 * (r + 2.0 * proj_points[..., 0] ** 2)
        tangential2 = p1 * (r + 2.0 * proj_points[..., 1] ** 2) + 2.0 * p2 * proj_points[..., 0] * proj_points[..., 1]

        proj_points[..., 0] = proj_points[..., 0] * radial.squeeze() + tangential1
        proj_points[..., 1] = proj_points[..., 1] * radial.squeeze() + tangential2

        # Apply camera matrix
        proj_points = ori_intrin.matmul(proj_points.unsqueeze(-1)).squeeze(-1)

        return proj_points

    def project_from_lidar_2_cam(self, points, rot_lidar2cam, tran_lidar2cam, ori_intrins, dist_coeffss,
                                 depth_thre=None):
        # ego_to_cam
        points = rot_lidar2cam.view(1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1) + tran_lidar2cam
        depths = points[..., 2:]
        mask = None
        if depth_thre is not None:
            mask = points[..., 2] > depth_thre

        points = torch.cat((points[..., :2] / depths, torch.ones_like(depths)), -1)

        # cam_to_img
        # points = intrins.view(1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)   
        ori_intrin = ori_intrins.view(1, 3, 3)
        dist_coeffs = dist_coeffss.view(1, 8)
        points = self.projectPoints(points, dist_coeffs, ori_intrin)
        # points = post_rots.view(1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        # points = points + post_trans.view(1, 3)
        # # points = points.view(B, N, Z, Y, X, 3)[..., :2]
        # points = points.view(-1,3).int().numpy()
        points[..., 2:] = depths
        return points, mask

    def get_rays_data(self, sce_id, recs, extrin_rot, extrin_tran, K_ori, distorts, H, W, aug_params,
                      align_first=False):
        rays_all = []
        raw_images = []
        aabb_min = self.bx2 - self.dx2 / 2
        aabb_max = self.bx2 - self.dx2 / 2 + self.dx2 * (self.nx2 - 1)
        T_lidar2cam = self.T_lidar2cam_list[sce_id]
        intrin = torch.Tensor(K_ori)
        rot = torch.Tensor(extrin_rot)
        tran = torch.Tensor(extrin_tran)
        image_name_base = None  # 防止数过大导致的精度损失
        first_pose = None
        pose_mats = torch.zeros((len(recs), 3, 4))
        for ii, sample in enumerate(recs):
            image_path, pose, vp = sample  # pose: world to lidar
            # print(image_path)
            rot_lidar2cam = T_lidar2cam[:3, :3]
            trans_lidar2cam = T_lidar2cam[:3, 3]
            rot_lidar2car = extrin_rot.dot(rot_lidar2cam)
            tran_lidar2car = extrin_rot.dot(trans_lidar2cam).T + extrin_tran
            rot_car2world = pose[:3, :3].dot(rot_lidar2car.getI())
            tran_car2world = pose[:3, :3].dot(tran_lidar2car.T) + pose[:3, 3]
            pose_mat = np.eye(4, dtype=np.float64)
            pose_mat[:3, :3] = rot_car2world
            pose_mat[:3, 3] = tran_car2world.T
            pose_mat = torch.DoubleTensor(pose_mat)

            if first_pose is None:
                first_pose = pose_mat
            pose2first_trans = torch.matmul(first_pose[0:3, 0:3].inverse(),
                                            pose_mat[0:3, 3] - first_pose[0:3, 3]).float()
            pose2first_rot = torch.matmul(first_pose[0:3, 0:3].inverse(), pose_mat[0:3, 0:3]).float()
            pose_mats[ii, 0:3, 0:3] = pose2first_rot
            pose_mats[ii, 0:3, 3] = pose2first_trans

            cam2car_normplane = torch.cat([rot, tran.reshape(-1, 1)], 1)
            cam2car = cam2car_normplane.clone()
            cam2car[:, 1:3] *= -1

            lidar2car = torch.cat([torch.Tensor(rot_lidar2car), torch.Tensor(tran_lidar2car).view(-1, 1)], 1)
            if align_first:
                cam2first_rot = torch.matmul(pose2first_rot, cam2car[:3, :3])
                cam2first_tran = torch.matmul(pose2first_rot, cam2car[:3, 3]) + pose2first_trans
                cam2car[:3, :3] = cam2first_rot
                cam2car[:3, 3] = cam2first_tran

                lidar2first_rot = torch.matmul(pose2first_rot, lidar2car[:3, :3])
                lidar2first_tran = torch.matmul(pose2first_rot, lidar2car[:3, 3]) + pose2first_trans
                lidar2car[:3, :3] = lidar2first_rot
                lidar2car[:3, 3] = lidar2first_tran

            img_name = float(image_path.split('/')[-1][:-4])
            if image_name_base is None:
                image_name_base = img_name
            ts_rel = img_name - image_name_base
            image = cv2.imread(image_path)
            depth_file = image_path.replace("org/img_ori", "depths_single_with_obj_time_ori").replace("/share/",
                                                                                                      "/share/occ_data/occ_data_ori/").replace(
                ".jpg", ".npy")
            with open(depth_file, 'rb') as f:
                try:
                    depth_org = np.load(f, allow_pickle=True)[:, :3]
                except ValueError:
                    print(f"np.load fail! depth_file:{depth_file}")
                    raise
                except:
                    print(f"np.load fail! depth_file:{depth_file}")
                    raise
            if 0:
                for ii in range(depth_org.shape[0]):
                    ptx = round(depth_org[ii, 0])
                    pty = round(depth_org[ii, 1])
                    cv2.circle(image, (ptx, pty), 3, (0, 0, 0), -1)
                cv2.imshow('image', image)
                cv2.waitKey(100)
                # cv2.imwrite('1.jpg', image)
            img_for_rays = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) / 255
            points_lidar = depth_org

            points_img, points_mask = self.project_from_lidar_2_cam(torch.Tensor(points_lidar).clone(),
                                                                    torch.Tensor(rot_lidar2cam),
                                                                    torch.Tensor(trans_lidar2cam).view(1, 3), intrin,
                                                                    torch.Tensor(distorts), 1)
            points_mask = points_mask * (points_img[..., 0] > 0) * (points_img[..., 1] > 0) * (
                    points_img[..., 0] < W) * (points_img[..., 1] < H)

            if 0:
                points_img_show = points_img[points_mask]
                for i in range(points_img_show.shape[0]):
                    cv2.circle(image, (int(points_img_show[i, 0]), int(points_img_show[i, 1])), 1, (0, 255, 0), -1)
                cv2.imshow('image', image)
                cv2.waitKey(100)
                # cv2.imwrite('1.jpg', image)

            min_idx = np.argmin(np.array(aug_params, dtype=object)[..., 0])
            resize, resize_dims, crop, flip, rotate = aug_params[min_idx]
            effect_crop = (np.array(crop) / resize).astype(np.int32)
            effect_crop = [max(0, effect_crop[0]), max(0, effect_crop[1]), min(W, effect_crop[2]),
                           min(H, effect_crop[3])]
            kept_depth = (points_img[:, 0] < effect_crop[2] - 1) * (points_img[:, 0] > effect_crop[0]) * (
                    points_img[:, 1] < effect_crop[3] - 1) * (points_img[:, 1] > effect_crop[1])
            points_mask = points_mask * kept_depth
            # depth_pre = depth_pre[np.where(kept_depth)]
            ### filter lidar points outside bev bbox
            ### depth_pre from pixel coordinate system
            # points_lidar = copy.deepcopy(depth_pre)

            # ### depth_pre from camera coordinate system
            # depths_2_car = np.array(lidar2car[:3,:3].numpy().dot(points_lidar).T + tran.numpy())
            # kept_depth2 = (depths_2_car[:, 0] >= self.grid_conf['xbound'][0]) * (depths_2_car[:, 0] <= (self.grid_conf['xbound'][1] - self.grid_conf['xbound'][2])) * (
            #             depths_2_car[:, 1] >= self.grid_conf['ybound'][0]) * (depths_2_car[:, 1] <= (self.grid_conf['ybound'][1] - self.grid_conf['ybound'][2])) * (
            #             depths_2_car[:, 2] >= self.grid_conf['zbound'][0]) * (depths_2_car[:, 2] <= (self.grid_conf['zbound'][1] - self.grid_conf['zbound'][2]))
            # depth_pre = depth_pre[np.where(kept_depth2)]
            ###
            points_lidar = torch.Tensor(points_lidar[np.where(points_mask)])
            ### debug
            if 0:
                pcd_cam = o3d.geometry.PointCloud()
                pcd_cam.points = o3d.utility.Vector3dVector(np.array(depths_2_cam.T))
                pcd_cam.colors = o3d.utility.Vector3dVector(
                    [[0, 255, 0] for i in range(np.array(depths_2_cam.T).shape[0])])

                pcd_car = o3d.geometry.PointCloud()
                pcd_car.points = o3d.utility.Vector3dVector(depths_2_car)
                pcd_car.colors = o3d.utility.Vector3dVector([[0, 0, 255] for i in range(depths_2_car.shape[0])])

                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name='pcd_car', width=352 * 3, height=128 * 3)
                vis.get_render_option().background_color = np.asarray([0.23, 0.64, 1])  # 设置一些渲染属性
                vis.add_geometry(pcd_cam, True)
                vis.add_geometry(pcd_car, True)

                view_control = vis.get_view_control()
                view_control.set_front([-1, 0, 0])
                view_control.set_lookat([10, 0, 2])
                view_control.set_up([0, 0, 1])
                view_control.set_zoom(0.025)
                view_control.rotate(0, 2100 / 40)

                vis.update_renderer()
                vis.run()
            ###

            depth_arr = torch.Tensor(points_img[np.where(points_mask)])
            depths_len = len(depth_arr)
            depths = depth_arr[:, 2].reshape(-1, 1)
            seg_file = image_path.replace("org/img_ori", "semantic_maps_ori").replace("/share/",
                                                                                      "/share/occ_data/occ_data_ori/").replace(
                ".jpg", ".png")
            if os.path.exists(seg_file):
                semantic_pre = cv2.imread(seg_file, 0)
                semantic_map = torch.Tensor(semantic_pre)
            else:
                semantic_map = torch.zeros(H, W, dtype=torch.int8)

            if self.is_train:
                """方案二: 根据depth的数量选取"""
                valid_crop_area = torch.zeros(H, W, dtype=torch.bool)
                valid_sky_area = torch.zeros(H, W, dtype=torch.bool)
                valid_crop_area[effect_crop[1]:effect_crop[3], effect_crop[0]:effect_crop[2]] = 1
                valid_sky_area[semantic_map == 2] = True
                sky_idxs = torch.nonzero((valid_sky_area * valid_crop_area) == True)

                if sky_idxs.size(0) > 0:
                    if sky_idxs.size(0) > depths_len:
                        sample_sky_idx = torch.randint(0, sky_idxs.size(0), (depths_len, 1))[:, 0]
                        sky_idxs = sky_idxs[sample_sky_idx, :]
                    # coords_x = torch.cat([depth_arr[:, 0].long(), sky_idxs[:, 1].long()], dim=0)
                    # coords_y = torch.cat([depth_arr[:, 1].long(), sky_idxs[:, 0].long()], dim=0)
                    coords_x = torch.cat([depth_arr[:, 0], sky_idxs[:, 1]], dim=0)
                    coords_y = torch.cat([depth_arr[:, 1], sky_idxs[:, 0]], dim=0)
                    depths = torch.cat([depths, torch.zeros(sky_idxs.size(0), 1)])
                else:
                    # coords_x = depth_arr[:, 0].long()
                    # coords_y = depth_arr[:, 1].long()
                    coords_x = depth_arr[:, 0]
                    coords_y = depth_arr[:, 1]

                # coords_x = torch.clamp(coords_x, 0, W - 1)
                # coords_y = torch.clamp(coords_y, 0, H - 1)
                coords_x_i = np.round(coords_x).long()
                coords_y_i = np.round(coords_y).long()
                coords_x_i = torch.clamp(coords_x_i, 0, W - 1)
                coords_y_i = torch.clamp(coords_y_i, 0, H - 1)
                coords = torch.cat([coords_x.reshape(-1, 1), coords_y.reshape(-1, 1)], dim=1)
                # print('max coorx:', torch.max(coords[..., 0]), 'min coorx:', torch.min(coords[..., 0]))
                # print('max coory:', torch.max(coords[..., 1]), 'min coory:', torch.min(coords[..., 1]))
                weights = torch.zeros(depths.shape)
                weights[:depths_len, 0] = 1
                img_for_rays = img_for_rays[coords_y_i, coords_x_i, :]
                rays_mask = semantic_map[coords_y_i, coords_x_i].reshape(-1, 1)
                near, far = 1.0 * torch.ones_like(depths), 120.0 * torch.ones_like(depths)

                # 训练时随机选取crop区域内self.sample_num个�
                # print('depth_file:', depth_file, depths.size(0))
                sample_idx = torch.randint(0, depths.size(0), (self.sample_num * 3, 1))[:, 0]
                lidar_idx = torch.where(sample_idx < depths_len)
                lidar_sample_idx = sample_idx[lidar_idx]
                rays_o_lidar, rays_d_lidar, depth_lidar = get_rays_from_lidar_points(points_lidar[lidar_sample_idx],
                                                                                     lidar2car)

                coords = coords[sample_idx, :]
                depths = depths[sample_idx, :]
                weights = weights[sample_idx, :]
                img_for_rays = img_for_rays[sample_idx, :]
                rays_mask = rays_mask[sample_idx, :]
                near = near[sample_idx, :]
                far = far[sample_idx, :]
                directions = get_ray_directions(H, W, K_ori, coords, self.is_train, distorts)
                rays_o, rays_d = get_rays(directions, cam2car)
                rays_d_for_depth = directions @ cam2car[:, :3].T
                dir_norm = torch.norm(rays_d_for_depth, dim=-1, keepdim=True)
                depths = depths * dir_norm

                rays_o[lidar_idx], rays_d[lidar_idx], depths[lidar_idx] = \
                    rays_o_lidar, rays_d_lidar, depth_lidar

                near_ins, far_ins, intersection_map = self.ray_box_intersection(np.array(rays_o), np.array(rays_d),
                                                                                near.numpy().copy(), far.numpy().copy(),
                                                                                aabb_min, aabb_max)
                near_ins, far_ins = torch.Tensor(near_ins).to(depths.device), torch.Tensor(far_ins).to(depths.device)
                near, far = torch.max(near_ins, near), far_ins

                depth_as_sky_idxs = torch.cat([torch.nonzero(depths <= near), torch.nonzero(depths >= far)])
                weights[depth_as_sky_idxs] = 0.0

                rays_mask[depth_as_sky_idxs] = get_label_id_mapping()["sky"]
                depths[depth_as_sky_idxs] = 0.0

                img_names = torch.Tensor([[img_name]] * near.size(0))
                ts_rels = torch.Tensor([[ts_rel]] * near.size(0))
                rays = torch.cat(
                    [
                        rays_o,
                        rays_d,
                        near,
                        far,
                        rays_mask,
                        depths,
                        weights,
                        ts_rels,
                        img_names,
                        img_for_rays  # rgb
                    ],
                    1,
                )  # (h*w, 14)
                valid_index = rays[:, 6] < rays[:, 7]
                rays = rays[valid_index]
                sample_idx = torch.randint(0, rays.size(0), (self.sample_num, 1))[:, 0]
                rays = rays[sample_idx]

                if vis_depth:
                    image_name = image_path.split('/')[-1][:-4]
                    print(f"saving... at samples/depth/{image_name}.ply")
                    os.makedirs("samples/depth", exist_ok=True)
                    pts = rays_o + rays_d * depths
                    gt_pcd = o3d.geometry.PointCloud()
                    gt_pcd.points = o3d.utility.Vector3dVector(pts.numpy())
                    o3d.io.write_point_cloud(
                        f"samples/depth/{image_name}.ply", gt_pcd
                    )
                    # np.save(f"samples/depth/{image_name}_dir_norm.npy", dir_norm.numpy())
            else:
                if 0:
                    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
                    grid = grid[effect_crop[1]:effect_crop[3], effect_crop[0]:effect_crop[2], :]
                    directions = get_ray_directions(H, W, K_ori, grid.reshape(-1, 2), True, distorts)
                    rays_o, rays_d = get_rays(directions, cam2car)

                    directions_for_depth = get_ray_directions(H, W, K_ori, depth_arr[:, :2], True, distorts)
                    rays_o_for_depth, rays_d_for_depth = get_rays(directions_for_depth, cam2car)

                    """generate pcd GT"""""
                    rays_d_for_depth2 = directions_for_depth @ cam2car[:, :3].T
                    dir_norm = torch.norm(rays_d_for_depth2, dim=-1, keepdim=True)
                    depths = depths * dir_norm
                    pts_gt = rays_o_for_depth + rays_d_for_depth * depths
                    # pts_gt = torch.cat([pts_gt,torch.zeros(rays_o.size(0)-depths_len,3)])
                    """generate pcd GT"""""

                    """求解rays与voxel的两个交点"""
                    near, far = 0. * torch.ones_like(directions_for_depth[..., :1]), 120.0 * torch.ones_like(
                        directions_for_depth[..., :1])
                    near_ins, far_ins, _ = self.ray_box_intersection(np.array(rays_o_for_depth),
                                                                     np.array(rays_d_for_depth), near.numpy().copy(),
                                                                     far.numpy().copy(), aabb_min, aabb_max)
                    near_ins, far_ins = torch.Tensor(near_ins).to(depths.device), torch.Tensor(far_ins).to(
                        depths.device)
                    near, far = near_ins, torch.min(depths + 3, far_ins)
                    img_names = torch.cat([torch.DoubleTensor([[img_name]]), torch.zeros(near.size(0) - 1, 1)])
                    ts_rels = torch.Tensor([[ts_rel]] * near.size(0))
                    rays = torch.cat(
                        [
                            rays_o_for_depth,
                            rays_d_for_depth,
                            near,
                            far,
                            ts_rels,
                            img_names,
                            depths,
                            pts_gt
                        ],
                        1,
                    )  # (h*w, 11)

                if 0:
                    select_num = 25000
                    directions_for_depth = get_ray_directions(H, W, K_ori, depth_arr[:, :2], True, distorts)
                    rays_o_for_depth, rays_d_for_depth = get_rays(directions_for_depth, cam2car)
                    rays_d_for_depth2 = directions_for_depth @ cam2car[:, :3].T
                    dir_norm = torch.norm(rays_d_for_depth2, dim=-1, keepdim=True)
                    depths = depths * dir_norm
                    pts_gt = rays_o_for_depth + rays_d_for_depth * depths
                    rays = torch.zeros((select_num, 13))
                    if pts_gt.size(0) > select_num:
                        rays[:, 10:13] = pts_gt[:select_num, :]
                    else:
                        rays[:pts_gt.size(0), 10:13] = pts_gt[:pts_gt.size(0), :]

                ### debug
                if 0:
                    sample_num = 20000
                    # trans_lidar2cam = [-0.000888, -0.668762, -1.201224]
                    trans_lidar2cam = np.array(T_lidar2cam[:3, 3]).reshape(-1)
                    rays_o_for_depth0, rays_d_for_depth0, depths0 = get_rays_from_lidar(K_ori, trans_lidar2cam, cam2car,
                                                                                        torch.cat([coords0, depths0],
                                                                                                  1), distorts)
                    z_val_lidar = torch.linspace(0.0, 1.0, 100) * depths[0:sample_num]
                    ray_lidar_o = (
                            rays_o_for_depth[0:sample_num].unsqueeze(-2) + rays_d_for_depth[0:sample_num].unsqueeze(
                        -2) * z_val_lidar.unsqueeze(-1)).view(-1, 3)
                    pcd_lidar_o = o3d.geometry.PointCloud()
                    pcd_lidar_o.points = o3d.utility.Vector3dVector(ray_lidar_o.cpu().numpy())
                    pcd_lidar_o.colors = o3d.utility.Vector3dVector([[0, 0, 255] for i in range(ray_lidar_o.shape[0])])

                    z_val_cam = torch.linspace(0.0, 1.0, 100) * depths_cam_o[0:sample_num]
                    ray_cam_o = (rays_o[0:sample_num].unsqueeze(-2) + rays_d[0:sample_num].unsqueeze(
                        -2) * z_val_cam.unsqueeze(-1)).view(-1, 3)
                    pcd_cam_o = o3d.geometry.PointCloud()
                    pcd_cam_o.points = o3d.utility.Vector3dVector(ray_cam_o.cpu().numpy())
                    pcd_cam_o.colors = o3d.utility.Vector3dVector([[0, 255, 0] for i in range(z_val_cam.shape[0])])

                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name='pts_gt', width=352 * 3, height=128 * 3)
                    vis.get_render_option().background_color = np.asarray([0.23, 0.64, 1])  # 设置一些渲染属性
                    vis.add_geometry(pcd_cam_o, True)
                    vis.add_geometry(pcd_lidar_o, True)

                    view_control = vis.get_view_control()
                    view_control.set_front([-1, 0, 0])
                    view_control.set_lookat([10, 0, 2])
                    view_control.set_up([0, 0, 1])
                    view_control.set_zoom(0.025)
                    view_control.rotate(0, 2100 / 40)

                    vis.update_renderer()
                    vis.run()

                # if rays.size(0) < select_num:
                #     rays = torch.cat([rays, torch.zeros(select_num - rays.size(0), 13)])
                # else:
                #     sample_idx = torch.randint(0, rays.size(0), (select_num, 1))[:, 0]
                #     rays = rays[sample_idx, :]
                ###
                if 1:
                    sample_num = 50000
                    rays_o_lidar, rays_d_lidar, depth_lidar = get_rays_from_lidar_points(points_lidar, lidar2car)

                    """generate pcd GT"""""
                    # rays_d_for_depth2 = directions_for_depth @ cam2car[:, :3].T
                    # dir_norm = torch.norm(rays_d_for_depth2, dim=-1, keepdim=True)
                    # depths = depths * dir_norm
                    pts_gt = rays_o_lidar + rays_d_lidar * depth_lidar

                    """求解rays与voxel的两个交点"""
                    near, far = 0. * torch.ones_like(depth_lidar), 120.0 * torch.ones_like(depth_lidar)
                    near_ins, far_ins, intersection_map = self.ray_box_intersection(np.array(rays_o_lidar),
                                                                                    np.array(rays_d_lidar),
                                                                                    near.numpy().copy(),
                                                                                    far.numpy().copy(), aabb_min,
                                                                                    aabb_max)
                    near_ins, far_ins = torch.Tensor(near_ins).to(depth_lidar.device), torch.Tensor(far_ins).to(
                        depth_lidar.device)
                    near, far = torch.max(near_ins, near), far_ins
                    # near, far = near_ins, torch.min(depths + 3, far_ins)

                    img_names = torch.DoubleTensor([[img_name]] * near.size(0))
                    ts_rels = torch.Tensor([[ts_rel]] * near.size(0))
                    rays_gt = torch.cat(
                        [
                            rays_o_lidar,
                            rays_d_lidar,
                            near,
                            far,
                            torch.zeros_like(far),
                            ts_rels,
                            img_names,
                            depth_lidar,
                            pts_gt
                        ],
                        1,
                    )  # (h*w, 11)
                    if rays_gt.size(0) < sample_num:
                        rays_gt = torch.cat([rays_gt, torch.zeros(sample_num - rays_gt.size(0), rays_gt.size(1))])
                    else:
                        sample_idx = torch.randint(0, rays_gt.size(0), (sample_num, 1))[:, 0]
                        rays_gt = rays_gt[sample_idx, :]

                    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
                    grid = grid[effect_crop[1]:effect_crop[3], effect_crop[0]:effect_crop[2], :]
                    semantic_map = semantic_map[effect_crop[1]:effect_crop[3], effect_crop[0]:effect_crop[2]]
                    # directions = get_ray_directions(H, W, K, grid.reshape(-1,2), True)
                    directions = get_ray_directions(H, W, K_ori, grid.reshape(-1, 2), True, distorts)
                    rays_o, rays_d = get_rays(directions, cam2car)

                    # gt_pcd = o3d.geometry.PointCloud()
                    # gt_pcd.points = o3d.utility.Vector3dVector(pts_gt.numpy())
                    # image_name = image_path.split('/')[-1][:-4]
                    # print(f"saving... at samples/depth/{image_name}.ply")
                    # os.makedirs("samples/depth", exist_ok=True)
                    # o3d.io.write_point_cloud(
                    #     f"samples/depth/{image_name}.ply", gt_pcd
                    # )

                    """求解rays与voxel的两个交点"""
                    near, far = 0. * torch.ones_like(directions[..., :1]), 120.0 * torch.ones_like(directions[..., :1])
                    near_ins, far_ins, intersection_map = self.ray_box_intersection(np.array(rays_o), np.array(rays_d),
                                                                                    near.numpy().copy(),
                                                                                    far.numpy().copy(), aabb_min,
                                                                                    aabb_max)
                    near_ins, far_ins = torch.Tensor(near_ins).to(depths.device), torch.Tensor(far_ins).to(
                        depths.device)
                    near, far = torch.max(near_ins, near), far_ins

                    img_names = torch.DoubleTensor([[img_name]] * near.size(0))
                    ts_rels = torch.Tensor([[ts_rel]] * near.size(0))
                    rays = torch.cat(
                        [
                            rays_o,
                            rays_d,
                            near,
                            far,
                            semantic_map.reshape(-1, 1),
                            ts_rels,
                            img_names,
                            torch.zeros_like(directions[..., :1]),
                            torch.zeros_like(rays_o)
                        ],
                        1,
                    )  # (h*w, 11)

                    rays = torch.cat([rays, rays_gt])

                    image = image[effect_crop[1]:effect_crop[3], effect_crop[0]:effect_crop[2]]
                    raw_images.append(normalize_img(image))

            rays_all.append(rays)
        if len(raw_images) > 0:
            raw_images = torch.stack(raw_images)
        else:
            raw_images = None
        return torch.stack(rays_all), pose_mats, raw_images

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

    def get_theta_mat(self, sce_id, recs, extrin_rot, extrin_tran):
        T_lidar2cam = self.T_lidar2cam_list[sce_id]
        theta_mat = np.zeros((len(recs), 3, 4))
        # theta_mat = np.zeros((len(recs), 2, 3))
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

            if ii > 0:
                pose_diff = pose_prev.getI().dot(pose_mat)

                # W, H, C = 120, 20, 6  # bev W-x  H-y  C-z
                # sx, sy, sz = W / 2, H / 2, C / 2
                # pre_mat = np.matrix([[0., 1. / sy, 0., 0.],
                #                      [1. / sx, 0., 0., -1.],
                #                      [0., 0., 1. / sz, -1. / sz],
                #                      [0., 0., 0., 1.]])
                #
                # post_mat = np.matrix([[0., sx, 0., sx],
                #                       [sy, 0., 0., 0.],
                #                       [0., 0., sz, sz/3.],
                #                       [0., 0., 0., 1.]])

                sx, sy, sz = (self.nx2 - 1) * self.dx2 * 0.5
                cx, cy, cz = self.bx2 + (self.nx2 * 0.5 - 1) * self.dx2
                # x = np.zeros(3)
                # x = (x-(self.bx2 - self.dx2/2))/(self.dx2*(self.nx2-1))*2-1
                pre_mat = np.matrix([[0., 1. / sy, 0., -cy / sy],
                                     [1. / sx, 0., 0., -cx / sx],
                                     [0., 0., 1. / sz, -cz / sz],
                                     [0., 0., 0., 1.]])

                post_mat = np.matrix([[0., sx, 0., cx],
                                      [sy, 0., 0., cy],
                                      [0., 0., sz, cz],
                                      [0., 0., 0., 1.]])
                theta_ = pre_mat.dot(pose_diff.dot(post_mat))[:3, :]  # shape = [3, 4]

                # yaw = math.atan2(pose_diff[1,0], pose_diff[0,0])
                # dx = pose_diff[0, 3]
                # dy = pose_diff[1, 3]
                #
                # yaw = yaw + np.random.uniform(-0.5*math.pi/180,0.5*math.pi/180)
                # dx = dx + np.random.uniform(-0.5,0.5)
                # dy = dy + np.random.uniform(-0.25,0.25)
                #
                # W, H = 120, 20  # bev W-x  H-y
                # cos = math.cos(yaw)
                # sin = math.sin(yaw)
                # sx = W/2
                # sy = H/2
                #
                # rel_pose = np.matrix([[cos, -sin, dx],
                #                     [sin, cos, dy],
                #                     [0., 0., 1.]])
                #
                # pre_mat = np.matrix([[0., 1./sy, 0.],
                #                     [1./sx, 0., -1.],
                #                     [0., 0., 1.]])
                #
                # post_mat = np.matrix([[0., sx, sx],
                #                     [sy, 0., 0.],
                #                     [0., 0., 1.]])
                # theta_ = pre_mat.dot(rel_pose.dot(post_mat))[:2, :]  #shape = [2,3]

                theta_mat[ii, :, :] = theta_

                ### debug
                if 0:
                    # pt0 = np.array([[0, -10, -2, 0], [119, 9.5, 3, 0], [30, 5, 1, 0], [59.5, -0.25, 0.5, 0]])
                    # origin = [0, -10, -2, 0]
                    # size = [119, 19.5, 5, 1]

                    pt0 = np.array([[0, -12, -3.5, 0], [95, 11, 3.5, 0], [47.5, -0.5, 0, 0], [23.25, -5.75, 1.75, 0]])
                    origin = [0, -12, -3.5, 0]
                    size = [95, 23, 7, 1]

                    pt0_f = (pt0 - origin) / size * 2 - 1
                    pt0_f[:, 3] = 1
                    pt0_f[:, [0, 1]] = pt0_f[:, [1, 0]]
                    pt = post_mat @ pt0_f[:, :].T
                    pt_inv = pre_mat @ pt

                    import torch.nn.functional as F
                    B, C, L, H, W = 8, 128, 6, 120, 40
                    theta0 = torch.eye(3, 4).view(1, 3, 4)
                    grids0 = F.affine_grid(theta0.expand(B, 3, 4), torch.Size((B, C, L, H, W)), align_corners=True)

                    theta1 = torch.Tensor(theta_).view(1, 3, 4)
                    grids1 = F.affine_grid(theta1.expand(B, 3, 4), torch.Size((B, C, L, H, W)),
                                           align_corners=True)  # 8,30,10,2 , theta - 8,2,3
                ###
            pose_prev = pose_mat
        return torch.Tensor(theta_mat)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""TFmapData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

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
        cams = self.choose_cams()
        # sub_ixes = []
        # for i in range(len(self.ixes)):
        #     if index >= self.ixes[i][0][0] and index < self.ixes[i][0][1]:
        #         sub_ixes = self.ixes[i][1]
        #         index -= self.ixes[i][0][0]
        #         break

        # index = 5000
        # sces_len = [(len(self.ixes[i][0]) - self.seq_len) for i in self.ixes.keys()]
        # scenes = [i for i in self.ixes.keys()]
        # sce_id = [i for i in range(len(sces_len)) if (sum(sces_len[:i]) <= index and sum(sces_len[:i+1]) > index)][0]
        # sce_id_ind = index - sum(sces_len[:sce_id])
        max_stride = min((len(self.ixes[sce_name][0]) - sce_id_ind) / self.seq_len, 5)

        # print('sce_id= ', sce_id, sce_id_ind, index)
        # max_stride = min(int((len(sub_ixes)-index)/self.seq_len), 2)
        stride = np.random.randint(1, max_stride + 1)

        if not self.is_train:
            stride = 3
        # stride = 10
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
        # axisangle_limits = [[-2./180.*np.pi, 2./180.*np.pi], [-3./180.*np.pi, 3./180.*np.pi], [-2./180.*np.pi, 2./180.*np.pi]]
        tran_limits = [[-2., 2.], [-1., 1.], [-1., 1.]]

        # runs1
        # axisangle_limits = [[-5./180.*np.pi, 5./180.*np.pi], [-5./180.*np.pi, 5./180.*np.pi], [-5./180.*np.pi, 5./180.*np.pi]]
        # tran_limits = [[-4., 4.], [-2., 2.], [-2., 2.]]

        axisangle_noises = [np.random.uniform(*angle_limit) for angle_limit in axisangle_limits]
        tran_noises = [np.random.uniform(*tran_limit) for tran_limit in tran_limits]

        if not self.is_train:
            tran_noises = np.zeros_like(tran_noises)
            axisangle_noises = np.zeros_like(axisangle_noises)

        noise_rot = euler_angles_to_rotation_matrix(axisangle_noises)
        noise_tran = np.array(tran_noises)
        extrin_rot = noise_rot.dot(rot)
        extrin_tran = noise_rot.dot(tran).T + noise_tran

        # imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data_tf(recs, cams, rot, tran, K)#extrin_rot, extrin_tran)
        imgs, rots, trans, intrins, dist_coeffss, post_rots, post_trans, cam_pos_embeddings, aug_params = self.get_image_data_tf(
            recs, cams, extrin_rot, extrin_tran, K_ori, distorts, H, W)

        # binimg = self.get_localmap_seg(sce_id, recs, rot, tran)#extrin_rot, extrin_tran)

        # binimg, valid_lf_areas = self.get_localmap_seg(sce_id, recs, extrin_rot, extrin_tran)
        # lf_label, lf_norm, lf_kappa, fork_patch, fork_scale, fork_offset, fork_ori = self.get_localmap_lf(sce_id, recs, rot, tran)
        # lf_label, lf_norm, lf_kappa, fork_patch, fork_scale, fork_offset, fork_ori = self.get_localmap_lf(sce_id, recs, extrin_rot, extrin_tran, valid_lf_areas)

        # cam_pos_embedding = self.get_cam_pos_embedding(self, recs, cams, intrins, distorts, post_trans, post_rots, 4)
        # fork_offset[..., 0] = 20.
        # print(fork_offset)
        rays, pose2first, raw_images = self.get_rays_data(sce_id, recs, extrin_rot, extrin_tran, K_ori, distorts, H, W,
                                                          aug_params, True)
        theta_mats = self.get_theta_mat(sce_id, recs, rot, tran)

        # 读取位姿数据
        pose_deltas, pose_mats = self.get_pose_mat(sce_id, recs, rot, tran)

        if self.is_train:
            return imgs, rots, trans, intrins, dist_coeffss, post_rots, post_trans, cam_pos_embeddings, rays, \
                theta_mats, pose_mats, pose2first
        else:
            return imgs, rots, trans, intrins, dist_coeffss, post_rots, post_trans, cam_pos_embeddings, rays, \
                theta_mats, pose_mats, pose2first, raw_images


class Segmentation1Data(TFmapData):
    def __init__(self, *args, **kwargs):
        super(Segmentation1Data, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        # index = self.data_idxs[index]
        # index = 1161
        sce_id, sce_id_ind = self.data_idxs[index]
        sce_name = self.image_paths_list[sce_id]
        cams = self.choose_cams()
        # sub_ixes = []

        # for i in range(len(self.ixes)):
        #     if index >= self.ixes[i][0][0] and index < self.ixes[i][0][1]:
        #         sub_ixes = self.ixes[i][1]
        #         index -= self.ixes[i][0][0]

        # sces_len = [(len(self.ixes[i][0]) - self.seq_len) for i in self.ixes.keys()]
        # scenes = [i for i in self.ixes.keys()]
        # sce_id = [i for i in range(len(sces_len)) if (sum(sces_len[:i]) <= index and sum(sces_len[:i+1]) > index)][0]
        # sce_id_ind = index - sum(sces_len[:sce_id])
        max_stride = min((len(self.ixes[sce_name][0]) - sce_id_ind) / self.seq_len, 5)

        # print('sce_id= ', sce_id, len(sces_len), index)
        stride = np.random.randint(1, max_stride + 1)
        stride = 1
        # recs = [sub_ixes[ii] for ii in range(index, index+self.seq_len*stride, stride)]
        recs = [self.ixes[sce_name][0][ii] for ii in range(sce_id_ind, sce_id_ind + self.seq_len * stride, stride)]
        rot = self.ixes[sce_name][1]
        tran = self.ixes[sce_name][2]
        K_ori = self.ixes[sce_name][3]
        distorts = self.ixes[sce_name][4]
        H = self.ixes[sce_name][5]
        W = self.ixes[sce_name][6]
        imgs, rots, trans, intrins, dist_coeffss, post_rots, post_trans, cam_pos_embeddings, aug_params = self.get_image_data_tf(
            recs, cams, rot, tran, K_ori, distorts, H, W)  # extrin_rot, extrin_tran)

        binimg, valid_lf_areas = self.get_localmap_seg(sce_id, recs, rot, tran)  # extrin_rot, extrin_tran)
        lf_label, lf_norm, lf_kappa, fork_patch, fork_scale, fork_offset, fork_ori = self.get_localmap_lf(sce_id, recs,
                                                                                                          rot, tran,
                                                                                                          valid_lf_areas)
        rays = self.get_rays_data(sce_id, recs, rot, tran, K_ori, distorts, H, W, aug_params)
        pose_mats = self.get_theta_mat(sce_id, recs, rot, tran)
        idx = torch.Tensor([index])
        img_paths = [rec[0] for rec in recs]

        return imgs, rots, trans, intrins, dist_coeffss, post_rots, post_trans, cam_pos_embeddings, binimg, lf_label, lf_norm, lf_kappa, fork_patch, fork_scale, fork_offset, fork_ori, idx, img_paths, rays, pose_mats


def worker_rnd_init(x):
    np.random.seed(13 + x)


def check_depth_exist(img_list):
    for img_path in img_list:
        depth_file = img_path.replace("org/img_ori", "depths_single_with_obj_time_ori").replace("/share/",
                                                                                                "/share/occ_data/occ_data_ori/").replace(
            ".jpg", ".npy")
        if not os.path.exists(depth_file):
            return False
    return True


def get_data_param(path_idxs, seq_len, is_train=True):
    data_infos_list = []
    param_infos_list = []
    vp_infos_list = []
    mesh_objs_list = []
    road_map_data_list = []
    roadmap_samples_list = []
    roadmap_forks_list = []
    map_paths_list = []
    ignore_map_paths_list = []
    ob_map_paths_list = []
    image_paths_list = []
    las_paths_list = []
    seg_paths_list = []
    idxs = []
    for sce_id, path in enumerate(path_idxs):
        #读取内参和位姿
        param_path = path + '/gen/param_infos.json'
        with open(param_path, 'r') as ff:
            param_infos = json.load(ff)
            param_infos_list.append(param_infos)

        geojson_path = path + '/org/geojson'
        map_list0 = sorted(glob.glob(geojson_path + '/标线标注*.geojson'))
        map_list1 = sorted(glob.glob(geojson_path + '/箭头标注*.geojson'))
        map_list2 = sorted(glob.glob(geojson_path + '/路沿标注*.geojson'))
        # map_list3 = sorted(glob.glob(geojson_path+'/问题路段范围*.geojson'))
        map_paths = map_list0 + map_list1 + map_list2
        # ignore_map_paths = map_list3
        map_paths_list.append(map_paths)
        # ignore_map_paths_list.append(ignore_map_paths)

        ignore_map_path = geojson_path + '/问题路段范围.geojson'
        if os.path.exists(ignore_map_path):
            ignore_map_paths_list.append(sorted(glob.glob(ignore_map_path)))
        else:
            ignore_map_paths_list.append([])

        ob_map_path = geojson_path + '/非可行驶区域.geojson'
        if os.path.exists(ob_map_path):
            ob_map_paths_list.append(sorted(glob.glob(ob_map_path)))
        else:
            ob_map_paths_list.append([])

        roadmap_path = path + '/org/hdmap'
        roadmap_list0 = sorted(glob.glob(roadmap_path + '/SideLines.geojson'))
        roadmap_list1 = sorted(glob.glob(roadmap_path + '/LaneLines.geojson'))
        road_map_data = {}
        for map_file in roadmap_list0:
            with open(map_file, encoding='utf-8') as fp:
                geojson = json.load(fp)
                print(geojson['name'])
                features = geojson['features']
                for ii, feature in enumerate(features):
                    poly = feature['geometry']['coordinates']
                    type = feature['geometry']['type']
                    if type == 'MultiLineString':
                        for pp in poly:
                            data = np.array(pp, dtype=np.float64)
                            print(type, data.shape)
                    elif type == "LineString":
                        llid = feature['properties']['LLID']
                        if llid is None:
                            continue
                        line_type = feature['properties']['LineType']
                        if line_type == 4:
                            continue
                        line_id = feature['properties']['Line_ID']
                        handside = feature['properties']['HandSide']
                        data = np.array(poly, dtype=np.float64)
                        if llid not in road_map_data:
                            road_map_data[llid] = {}
                            road_map_data[llid]["leftside"] = []
                            road_map_data[llid]["rightside"] = []

                        if handside == 1:
                            road_map_data[llid]["leftside"].append([line_id, data])
                        elif handside == 2:
                            road_map_data[llid]["rightside"].append([line_id, data])

                        # print(type, data.shape)
                    # elif type == "Polygon":
                    #     print(type)
                    # elif type == "MultiPolygon":
                    #     print(type)
                    # for pp in poly:
                    #    data = np.array(pp, dtype=np.float64)
                    # print(type, data.shape)
                    # all_map_data.append(data)
                    # elif type == "Point":
                    #     print(type)
                    # else:
                    #     print(type)

        for map_file in roadmap_list1:
            with open(map_file, encoding='utf-8') as fp:
                geojson = json.load(fp)
                print(geojson['name'])
                features = geojson['features']
                for ii, feature in enumerate(features):
                    poly = feature['geometry']['coordinates']
                    type = feature['geometry']['type']
                    if type == 'MultiLineString':
                        for pp in poly:
                            data = np.array(pp, dtype=np.float64)
                            print(type, data.shape)
                    elif type == "LineString":
                        llid = int(feature['properties']['LLID'])
                        if llid is None:
                            continue
                        if llid not in road_map_data:
                            road_map_data[llid] = {}
                            road_map_data[llid]["leftside"] = []
                            road_map_data[llid]["rightside"] = []

                        data = np.array(poly, dtype=np.float64)
                        road_map_data[llid]['cl'] = data
                        # print(type, data.shape)
                    # elif type == "Polygon":
                    #     print(type)
                    # elif type == "MultiPolygon":
                    #     print(type)
                    # for pp in poly:
                    #    data = np.array(pp, dtype=np.float64)
                    # print(type, data.shape)
                    # all_map_data.append(data)
                    # elif type == "Point":
                    #     print(type)
                    # else:
                    #     print(type)
        road_map_data_list.append(road_map_data)

        roadmap_sample_path = path + '/gen/samples'
        roadmap_samples = {}
        for key in road_map_data.keys():
            left_name = roadmap_sample_path + '/' + str(key) + "_l.npy"
            right_name = roadmap_sample_path + '/' + str(key) + "_r.npy"
            roadmap_samples[key] = {}
            if os.path.exists(left_name):
                roadmap_samples[key]["left"] = left_name
            if os.path.exists(right_name):
                roadmap_samples[key]["right"] = right_name
        roadmap_samples_list.append(roadmap_samples)

        fork_path = path + '/gen/lf_fork.json'
        fork_sample_path = path + '/gen/fork_samples'
        roadmap_forks = {}
        with open(fork_path, "r") as ff:
            forks = json.load(ff)
            roadmap_idxs = []
            for key in forks.keys():
                key0, key1 = key.split("_")
                key0 = int(key0)
                key1 = int(key1)
                roadmap_idxs.append(key0)
                roadmap_idxs.append(key1)
                re1 = os.path.exists(fork_sample_path + '/' + key + "_idxs.npy")
                re2 = os.path.exists(fork_sample_path + '/' + key + "_pts.npy")
                if re1 and re2:
                    # fork_sample_idxs = np.load(fork_sample_path+'/'+key+"_idxs.npy")
                    with open(fork_sample_path + '/' + key + "_idxs.npy", 'rb') as f:
                        fork_sample_idxs = np.load(f, allow_pickle=True)
                    # fork_sample_pts = np.load(fork_sample_path+'/'+key+"_pts.npy")
                    with open(fork_sample_path + '/' + key + "_pts.npy", 'rb') as f:
                        fork_sample_pts = np.load(f, allow_pickle=True)
                    roadmap_forks[key] = {}
                    roadmap_forks[key]['area'] = forks[key]
                    roadmap_forks[key]['idxs'] = fork_sample_idxs
                    roadmap_forks[key]['pts'] = fork_sample_pts
            roadmap_forks_list.append(roadmap_forks)

        obj_list = sorted(glob.glob(path + '/org/mesh/*.obj'))
        mesh_objs = []
        for ii, obj_file in enumerate(obj_list):
            obj_mesh = o3d.io.read_triangle_mesh(obj_file)
            if np.asarray(obj_mesh.vertices).shape[0] == 0:
                continue
            mesh_objs.append(obj_mesh)
        mesh_objs_list.append(mesh_objs)

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
                #  idxs += [[sce_id, int(item)] for item in lines]
                ### delete img without depth file
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
    return param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list, road_map_data_list, roadmap_samples_list, \
        roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list, image_paths_list, las_paths_list, \
        seg_paths_list, idxs


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz, seq_len, sample_num,
                 nworkers, parser_name, datatype, use_train=True):
    parser = {
        'segmentationdata': SegmentationData,
        'segmentation1data': Segmentation1Data,
    }[parser_name]

    train_datasets = ['bev_gnd/20230224_20km/20230306_1.56km_ZJ245', 'bev_gnd/20221130_100km/20230105_4.47km_ZJ139',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0217_4',
                      'bev_gnd/20221121_50km/20221213_7.21km_ZJ089',
                      'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0217_2',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0301_2',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0224_1',
                      'bev_gnd/20221122_120km/20221214_4.55km_ZJ095',
                      'bev_gnd/20221130_100km/20221226_4.64km_ZJ136', 'bev_gnd/20230208_30km/20230216_2.11km_ZJ227',
                      'bev_gnd/20230228_75km/20230403_1.55km_ZJ271', 'bev_gnd/20221122_120km/20221215_4.65km_ZJ105',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0217_5',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0223_4',
                      'bev_gnd/20221121_50km/20221212_7.24km_ZJ087',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0301_7',
                      'bev_gnd/20221206_80km/20230315_4.53km_ZJ169', 'bev_gnd/20221122_120km/20221219_4.31km_ZJ120',
                      'bev_gnd/20221122_120km/20221223_4.35km_ZJ123',
                      'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0217_5',
                      'bev_gnd/20221122_120km/20221216_4.71km_ZJ110', 'bev_gnd/20221122_120km/20221215_4.70km_ZJ104',
                      'bev_gnd/20221122_120km/20221219_4.52km_ZJ118',
                      'bev_gnd/20221110_75km/20221130_6.93km_ZJ065', 'bev_gnd/20221201_100km/20230322_4.72km_ZJ164',
                      'bev_gnd/20221201_100km/20230104_4.67km_ZJ158', 'bev_gnd/20221110_75km/20221202_6.23km_ZJ069',
                      'bev_gnd/20221110_75km/20221125_6.94km_ZJ063',
                      'bev_gnd/20221206_80km/20230406_4.18km_ZJ172', 'bev_gnd/20221201_100km/20230103_4.34km_ZJ155',
                      'bev_gnd/20221122_120km/20221221_4.49km_ZJ122', 'bev_gnd/20221121_50km/20221212_7.28km_ZJ084',
                      'bev_gnd/20221121_50km/20221212_7.29km_ZJ085',
                      'bev_gnd/20221116_80km/20221208_6.96km_ZJ081', 'bev_gnd/20221110_75km/20221129_6.60km_ZJ062',
                      'bev_gnd/20230208_30km/20230223_3.54km_ZJ228',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0224_2',
                      'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0217_7',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0301_3',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0301_9',
                      'bev_gnd/20221122_120km/20221220_4.11km_ZJ126', 'bev_gnd/20221110_75km/20221129_6.93km_ZJ064',
                      'bev_gnd/20230224_20km/20230309_4.00km_ZJ250', 'bev_gnd/20221130_100km/20221223_4.46km_ZJ133',
                      'bev_gnd/20221130_100km/20221227_4.70km_ZJ138', 'bev_gnd/20221201_100km/20230105_4.72km_ZJ162',
                      'bev_gnd/20221122_120km/20221219_4.46km_ZJ121',
                      'bev_gnd/20221122_120km/20221214_4.56km_ZJ096', 'bev_gnd/20221206_80km/20230410_4.43km_ZJ174',
                      'bev_gnd/20221116_80km/20221201_6.58km_ZJ071',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0224_4',
                      'bev_gnd/20221122_120km/20221219_4.46km_ZJ114',
                      'bev_gnd/20221116_80km/20221208_7.12km_ZJ080', 'bev_gnd/20221130_100km/20221222_4.69km_ZJ134',
                      'bev_gnd/20221122_120km/20221215_4.70km_ZJ103',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0224_3',
                      'bev_gnd/20221110_75km/20221130_6.92km_ZJ066',
                      'bev_gnd/20221116_80km/20221206_7.07km_ZJ076', 'bev_gnd/20221201_100km/20230104_4.68km_ZJ159',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0301_1',
                      'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0223_4',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0224_2',
                      'bev_gnd/20221121_50km/20221212_3.64km_ZJ086', 'bev_gnd/20221122_120km/20221219_4.52km_ZJ115',
                      'bev_gnd/20221116_80km/20221207_7.14km_ZJ079', 'bev_gnd/20221130_100km/20221226_4.70km_ZJ135',
                      'bev_gnd/20221130_100km/20221222_4.57km_ZJ131',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0223_4',
                      'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0223_1',
                      'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0301_5',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0301_2',
                      'bev_gnd/20221122_120km/20221219_4.44km_ZJ119',
                      'bev_gnd/20221130_100km/20221227_4.66km_ZJ140', 'bev_gnd/20221130_100km/20221227_4.63km_ZJ141',
                      'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0223_3',
                      'bev_gnd/20221116_80km/20221202_6.98km_ZJ072', 'bev_gnd/20221116_80km/20221206_7.06km_ZJ077',
                      'bev_gnd/20221206_80km/20230315_3.53km_ZJ168',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0301_5',
                      'bev_gnd/20230228_75km/20230403_1.88km_ZJ270',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0301_3',
                      'bev_gnd/20221122_120km/20221214_4.58km_ZJ094',
                      'bev_gnd/20221122_120km/20221216_4.68km_ZJ109', 'bev_gnd/20221201_100km/20230322_4.44km_ZJ165',
                      'bev_gnd/20221121_50km/20221213_7.28km_ZJ088', 'bev_gnd/20221110_75km/20221201_6.59km_ZJ068',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0301_5',
                      'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0224_1',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0301_6',
                      'bev_gnd/20221201_100km/20230317_1.42km_ZJ167', 'bev_gnd/20221201_100km/20230317_4.50km_ZJ166',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0217_2',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0224_3',
                      'bev_gnd/20221116_80km/20221207_7.04km_ZJ078', 'bev_gnd/20221122_120km/20221220_4.47km_ZJ125',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0217_3',
                      'bev_gnd/20221130_100km/20221220_4.64km_ZJ130', 'bev_gnd/20221201_100km/20230103_4.45km_ZJ156',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0224_4',
                      'bev_gnd/20221122_120km/20221216_3.79km_ZJ113',
                      'bev_gnd/20221110_75km/20221130_6.60km_ZJ067', 'bev_gnd/20221201_100km/20230104_4.73km_ZJ161',
                      'bev_gnd/20221116_80km/20221201_2.93km_ZJ082', 'bev_gnd/20221121_50km/20221213_7.23km_ZJ090',
                      'bev_gnd/20221130_100km/20221222_4.63km_ZJ132',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0224_1',
                      'bev_gnd/20221130_100km/20221228_4.37km_ZJ142', 'bev_gnd/20221121_50km/20221209_7.36km_ZJ083',
                      'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0224_3',
                      'bev_gnd/20221201_100km/20230104_4.69km_ZJ160',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0217_7',
                      'bev_gnd/20221116_80km/20221205_7.05km_ZJ073', 'bev_gnd/20221122_120km/20221219_4.49km_ZJ117',
                      'bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0301_8',
                      'bev_gnd/20221201_100km/20230109_4.64km_ZJ163',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0217_6',
                      'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0224_2',
                      'bev_gnd/20221122_120km/20221220_4.50km_ZJ127',
                      'bev_gnd_fucai/20230208_30km/20230222_4.43km_ZJ229_0301_6', ]
    train_datasets += ['bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0217_3',
                       'bev_gnd_fucai/20220919_100km/20220930_3km_done_2022-10-07-14-27-34',
                       'bev_gnd_fucai/20221129_cexiao/20221020_5.78km_2022-11-29-13-55-40',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0301_2',
                       'bev_gnd_fucai/20220919_100km/20221008_6.7km_done_2022-10-07-14-47-42',
                       'bev_gnd_fucai/20221116_80km/20221202_6.98km_1119_1',
                       'bev_gnd_fucai/20230224_20km/20230306_1.56km_ZJ245_0301_5',
                       'bev_gnd_fucai/20220919_100km/20221009_5.96km_20221009-1_2022-10-07-10-15-19',
                       'bev_gnd_fucai/20220919_100km/20220929_5.65km_20220929_2_2022-10-03-09-42-26',
                       'bev_gnd_fucai/20221129_cexiao/20221026_6.78km_2022-11-30-11-19-19',
                       'bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0217_5',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0301_4',
                       'bev_gnd_fucai/20220919_100km/20221018_5.16km_2022-10-01-14-54-37-5.16km',
                       'bev_gnd_fucai/20221020_50km/20221114_4.11km_2022-11-12-11-47-43',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0223_4',
                       'bev_gnd_fucai/20221018_13km/20221021_5.98km_2022-10-22-15-30-39',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0223_2',
                       'bev_gnd_fucai/20220919_100km/20220927_5.15km_2022-10-01-13-44-09-20220927-5.15km',
                       'bev_gnd_fucai/20230224_20km/20230307_4.08km_ZJ247_part1_0217_6',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-02-12-30-53',
                       'bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0223_2',
                       'bev_gnd_fucai/20230208_30km/20230223_0.63km_ZJ233_0223_4',
                       'bev_gnd_fucai/20221129_cexiao/20221031_5.62km_2022-12-03-14-52-59',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0217_6',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0301_1',
                       'bev_gnd_fucai/20220919_100km/20221018_5.16km_2022-10-07-15-33-00_part1',
                       'bev_gnd_fucai/20230224_20km/20230306_1.56km_ZJ245_0301_3',
                       'bev_gnd_fucai/20220919_100km/20220927_5.15km_2022-10-07-10-40-29',
                       'bev_gnd_fucai/20230208_30km/20230223_0.63km_ZJ233_0217_2',
                       'bev_gnd_fucai/20221018_13km/20221020_5.78km_2022-10-25-11-42-01',
                       'bev_gnd_fucai/20230224_20km/20230307_4.41km_ZJ248_0217_6',
                       'bev_gnd_fucai/20221129_cexiao/20221031_5.62km_2022-12-03-08-38-59',
                       'bev_gnd_fucai/20221020_50km/20221031_5.62km_2022-11-12-13-08-15',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0217_2_part_1',
                       'bev_gnd_fucai/20221129_cexiao/20221026_6.78km_2022-11-30-10-03-27',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-01-16-08-29',
                       'bev_gnd_fucai/20220919_100km/20221010_6.52km_done_2022-10-03-10-12-38-20221010-6.52km',
                       'bev_gnd_fucai/20220919_100km/20220923_3km_2022-10-06-10-50-48',
                       'bev_gnd_fucai/20221129_cexiao/20221026_6.78km_2022-11-30-08-34-29',
                       'bev_gnd_fucai/20230208_30km/20230223_0.63km_ZJ233_0301_3',
                       'bev_gnd_fucai/20221020_50km/20221031_5.62km_2022-11-12-16-40-02',
                       'bev_gnd_fucai/20220919_100km/20220927_5.15km_2022-10-01-10-12-41-20220927',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0217_6',
                       'bev_gnd_fucai/20220919_100km/20220930_3km_done_2022-10-01-10-02-37-20220930',
                       'bev_gnd_fucai/20220919_100km/20220923_3km_2022-10-05-14-34-52',
                       'bev_gnd_fucai/20220919_100km/20221010_6.52km_done_2022-10-06-11-15-58-P2',
                       'bev_gnd_fucai/20220919_100km/20220928_6.02km_2022-10-06-10-35-42',
                       'bev_gnd_fucai/20220919_100km/20221013_4.28km_2022-10-06-11-36-06',
                       'bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0217_4',
                       'bev_gnd_fucai/20220919_100km/20221013_4.28km_2022-10-07-15-12-52',
                       'bev_gnd_fucai/20221129_cexiao/20221026_6.78km_2022-11-30-12-40-21',
                       'bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0217_7',
                       'bev_gnd_fucai/20230208_30km/20230223_0.63km_ZJ233_0301_4',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0223_1',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-01-15-31-33',
                       'bev_gnd_fucai/20221129_cexiao/20221026_6.78km_2022-11-30-07-42-44',
                       'bev_gnd_fucai/20220919_100km/20221017_6.22km_2022-10-07-11-25-47',
                       'bev_gnd_fucai/20221020_50km/20221114_4.11km_2022-10-31-16-00-32',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-02-07-35-08',
                       'bev_gnd_fucai/20220919_100km/20220927_5.15km_2022-10-06-11-05-54',
                       'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0217_4',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-02-15-23-50',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0217_1',
                       'bev_gnd_fucai/20220919_100km/20221009_5.96km_20221009-1_2022-10-06-10-40-44',
                       'bev_gnd_fucai/20221129_cexiao/20221026_6.78km_2022-11-30-12-01-07',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0223_4',
                       'bev_gnd_fucai/20220919_100km/20220927_5.15km_2022-10-03-09-57-32-20220927-5.15km',
                       'bev_gnd_fucai/20220919_100km/20221018_5.16km_2022-10-07-15-33-00_part2',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0301_4',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0223_1',
                       'bev_gnd_fucai/20221020_50km/20221031_2.94km_2022-11-12-16-24-56',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0223_3',
                       'bev_gnd_fucai/20220919_100km/20220929_5.65km_20220929_2_2022-10-01-13-29-03',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0217_1',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0217_4',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0217_5',
                       'bev_gnd_fucai/20220919_100km/20220929_5.65km_20220929_1_2022-10-07-14-22-32',
                       'bev_gnd_fucai/20220919_100km/20221012_4.87km_2022-10-06-11-31-04',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0217_2_part_2',
                       'bev_gnd_fucai/20220919_100km/20220929_5.65km_20220929_2_2022-10-07-10-25-23',
                       'bev_gnd_fucai/20230208_30km/20230223_0.63km_ZJ233_0301_5',
                       'bev_gnd_fucai/20230224_20km/20230307_4.08km_ZJ247_part1_0217_1',
                       'bev_gnd_fucai/20221129_cexiao/20221020_5.78km_2022-11-29-14-44-42',
                       'bev_gnd_fucai/20221129_cexiao/20221031_5.62km_2022-12-03-08-59-00',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0223_2',
                       'bev_gnd_fucai/20220919_100km/20220928_6.02km_2022-10-07-14-12-28',
                       'bev_gnd_fucai/20220919_100km/20221010_6.52km_done_2022-10-07-14-47-42_part1',
                       'bev_gnd_fucai/20221129_cexiao/20221031_5.62km_2022-12-03-10-26-31',
                       'bev_gnd_fucai/20221129_cexiao/20221031_5.62km_2022-12-03-14-29-22',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0217_3',
                       'bev_gnd_fucai/20220919_100km/20220929_5.65km_20220929_1_2022-10-01-13-24-01',
                       'bev_gnd_fucai/20221018_13km/20221020_5.78km_2022-10-27-13-23-31',
                       'bev_gnd_fucai/20230208_30km/20230223_0.63km_ZJ233_0217_1',
                       'bev_gnd_fucai/20220919_100km/20221018_5.16km_2022-10-03-11-02-58-5.16km',
                       'bev_gnd_fucai/20220919_100km/20221018_5.16km_2022-10-07-15-33-00_part3',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-02-11-51-40',
                       'bev_gnd_fucai/20220919_100km/20221013_4.28km_2022-10-07-11-15-43',
                       'bev_gnd_fucai/20220919_100km/20220928_6.02km_2022-10-01-13-13-57-6.02km',
                       'bev_gnd_fucai/20221129_cexiao/20221031_5.62km_2022-12-03-08-19-09',
                       'bev_gnd_fucai/20220919_100km/20220930_3km_done_2022-10-01-13-34-05-20220930-3km_1_done',
                       'bev_gnd_fucai/20221129_cexiao/20221031_5.62km_2022-12-03-11-18-02',
                       'bev_gnd_fucai/20220919_100km/20221017_6.22km_2022-10-03-10-52-54-6.22km',
                       'bev_gnd_fucai/20221129_cexiao/20221031_5.62km_2022-12-03-15-36-22',
                       'bev_gnd_fucai/20221129_cexiao/20221031_5.62km_2022-12-03-16-33-14',
                       'bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0223_1',
                       'bev_gnd_fucai/20221129_cexiao/20221020_5.78km_2022-11-29-15-34-28_part1',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-02-08-22-10',
                       'bev_gnd_fucai/20221129_cexiao/20221020_5.78km_2022-11-29-15-34-28_part2',
                       'bev_gnd_fucai/20220919_100km/20220929_5.65km_20220929_1_2022-10-06-10-45-46',
                       'bev_gnd_fucai/20221020_50km/20221028_6.25km_2022-11-12-16-29-58',
                       'bev_gnd_fucai/20221129_cexiao/20221031_5.62km_2022-12-03-07-53-22',
                       'bev_gnd_fucai/20230208_30km/20230223_0.63km_ZJ233_0217_6',
                       'bev_gnd_fucai/20220919_100km/20220927_5.15km_2022-10-08-22-12-29',
                       'bev_gnd_fucai/20220919_100km/20221014_4.84km_20221014-1_2022-10-06-11-41-08',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-02-13-16-10',
                       'bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0223_3',
                       'bev_gnd_fucai/20220919_100km/20220929_5.65km_20220929_1_2022-10-03-09-37-24',
                       'bev_gnd_fucai/20220919_100km/20220929_5.65km_20220929_2_2022-10-06-10-50-48',
                       'bev_gnd_fucai/20221129_cexiao/20221026_6.78km_2022-11-30-09-22-22',
                       'bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0217_2',
                       'bev_gnd_fucai/20221018_13km/20221020_5.78km_2022-10-24-12-29-12',
                       'bev_gnd_fucai/20220919_100km/20221010_6.52km_done_2022-10-06-11-15-58-P1',
                       'bev_gnd_fucai/20220919_100km/20220923_3km_2022-10-01-13-34-05_done',
                       'bev_gnd_fucai/20220919_100km/20220923_3km_2022-10-03-09-47-28',
                       'bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0217_1',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-02-09-13-20',
                       'bev_gnd_fucai/20221020_50km/20221031_2.94km_2022-10-31-17-11-00',
                       'bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0217_6',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-02-16-08-41',
                       'bev_gnd_fucai/20230208_30km/20230223_3.54km_ZJ228_0301_1',
                       'bev_gnd_fucai/20221018_13km/20221020_5.78km_2022-10-24-15-52-11',
                       'bev_gnd_fucai/20221129_cexiao/20221028_6.25km_2022-12-02-10-30-18',
                       'bev_gnd_fucai/20220919_100km/20220923_3km_2022-10-01-10-02-37-1',
                       'bev_gnd_fucai/20230208_30km/20230223_4.38km_ZJ231_0301_2',
                       'bev_gnd_fucai/20230208_30km/20230221_4.48km_ZJ230_0301_5',
                       'bev_gnd_fucai/20221018_13km/20221020_5.78km_2022-10-22-15-40-43-part2',
                       'bev_gnd_fucai/20220919_100km/20220923_3km_2022-10-05-10-07-51',
                       'bev_gnd_fucai/20230208_30km/20230223_2.07km_ZJ235_0223_3',
                       'bev_gnd_fucai/20220919_100km/20221009_5.96km_20221009-1_2022-10-01-09-52-33-20221009-1', ]
    train_datasets += ['bev_gnd_fucai/20221129_cexiao/20221027_6.58km_2022-12-04-11-38-23_part1',
                       'bev_gnd_fucai/20221129_cexiao/20221027_6.58km_2022-12-04-09-08-14',
                       'bev_gnd_fucai/20221129_cexiao/20221027_6.58km_2022-12-04-11-38-23_part2',
                       'bev_gnd_fucai/20221129_cexiao/20221027_6.58km_2022-12-04-08-37-08',
                       ]
    val_datasets = ['bev_gnd_fucai/20221129_cexiao/20221027_6.58km_2022-12-04-08-37-08']
    val_datasets = ['bev_gnd_fucai/20220919_100km/20221008_6.7km_done_2022-10-03-10-07-36-20221008-6.7km']
    # val_datasets = ['bev_gnd/20230224_20km/20230306_1.56km_ZJ245']
    # val_datasets = ['bev_gnd_fucai/20230208_30km/20230216_2.11km_ZJ227_0301_4']
    # val_datasets = ['bev_gnd_fucai/20221129_cexiao/20221020_5.78km_2022-11-29-13-55-40']
    if not use_train:
        train_datasets = val_datasets

    # train_datasets = val_datasets

    # train_datasets = ['bev_gnd_fucai/20230224_20km/20230306_1.56km_ZJ245_0301_3'] ### error
    # val_datasets = ['bev_gnd/20230224_20km/20230309_4.00km_ZJ250']

    train_list = [dataroot + '/' + dataset for dataset in train_datasets]

    # train_list += [dataroot + '/bev_data_8M/' + dataset for dataset in train_datasets_8M]

    # print('train_list = ', train_list)

    val_list = [dataroot + '/' + dataset for dataset in val_datasets]
    # train_list = data_list[0:14]
    # train_list = data_list[0:5]
    # val_list = data_list[14:17]
    train_idxs = []
    val_idxs = []

    # lines = [fcode.strip().split() for fcode in open("/zhangjingjuan/NeRF/bev_osr_distort/data/used_fcode.txt", 'r').readlines()]
    used_fcodes = {}
    # for line in lines:
    #     used_fcodes[line[0]] = int(line[1])
    trainloader = None
    train_sampler = None
    if use_train:
        # 读取参数信息
        param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list, road_map_data_list, roadmap_samples_list, \
            roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list, image_paths_list, las_paths_list, \
            seg_paths_list, train_idxs = get_data_param(
            train_list, seq_len, is_train=True)

        traindata = parser(train_idxs, param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list,
                           road_map_data_list, roadmap_samples_list, roadmap_forks_list, map_paths_list,
                           ignore_map_paths_list, ob_map_paths_list, image_paths_list, las_paths_list, seg_paths_list,
                           is_train=True, data_aug_conf=data_aug_conf,
                           grid_conf=grid_conf, seq_len=seq_len, used_fcodes=used_fcodes, sample_num=sample_num,
                           datatype=datatype, crop_size=96)

        train_sampler = DistributedSampler(traindata)

        if parser_name == "segmentation1data":
            print("segmentation1data!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                                      shuffle=False,
                                                      num_workers=nworkers,
                                                      pin_memory=False, persistent_workers=True)
        else:
            trainloader = torch.utils.data.DataLoader(traindata, sampler=train_sampler, batch_size=bsz,
                                                      shuffle=False,
                                                      num_workers=nworkers,
                                                      drop_last=True,
                                                      worker_init_fn=worker_rnd_init,
                                                      pin_memory=False, prefetch_factor=32)

    param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list, road_map_data_list, roadmap_samples_list, \
        roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list, image_paths_list, las_paths_list, \
        seg_paths_list, val_idxs = get_data_param(
        val_list, seq_len, is_train=False)

    valdata = parser(val_idxs, param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list, road_map_data_list,
                     roadmap_samples_list, roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list,
                     image_paths_list, las_paths_list, seg_paths_list, is_train=False, data_aug_conf=data_aug_conf,
                     grid_conf=grid_conf, seq_len=seq_len, used_fcodes=used_fcodes, sample_num=sample_num,
                     datatype=datatype, crop_size=96)

    val_sampler = DistributedSampler(valdata)
    # sampler = val_sampler, 

    # valloader = torch.utils.data.DataLoader(valdata, sampler = val_sampler,  batch_size=bsz,
    #                                         shuffle=False,
    #                                         drop_last=True,
    #                                         num_workers=nworkers,
    #                                         pin_memory=True, persistent_workers=True)
    # valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
    #                                         shuffle=False,
    #                                         num_workers=nworkers,
    #                                         pin_memory=False, persistent_workers=True)
    # trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
    #                                         shuffle=False,
    #                                         num_workers=0,
    #                                         pin_memory=True, persistent_workers=False)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True, persistent_workers=False)

    return train_sampler, val_sampler, trainloader, valloader
