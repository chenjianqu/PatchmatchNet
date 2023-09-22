import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R


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

