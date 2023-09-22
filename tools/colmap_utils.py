import argparse
import cv2
import numpy as np
import os
import shutil
import struct
from typing import Dict, List, NamedTuple, Tuple


# ============================ read_model.py ============================#
class CameraModel(NamedTuple):
    model_id: int
    model_name: str
    num_params: int


class Camera(NamedTuple):
    id: int
    model: str
    width: int
    height: int
    params: List[float]


class Image(NamedTuple):
    id: int
    qvec: List[float]
    tvec: List[float]
    camera_id: int
    name: str
    point3d_ids: List[int] = []


class Point3D(NamedTuple):
    id: int
    xyz: List[float]
    rgb: List[int]
    error: float
    image_ids: List[int]
    point2d_ids: List[int]


CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str) -> Tuple:
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


def read_cameras_text(path: str) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    model_cameras: Dict[int, Camera] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                cam_id = int(elements[0])
                model = elements[1]
                width = int(elements[2])
                height = int(elements[3])
                params = list(map(float, elements[4:]))
                model_cameras[cam_id] = Camera(cam_id, model, width, height, params)
    return model_cameras


def read_cameras_binary(path: str) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    model_cameras: Dict[int, Camera] = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        print("num of cameras")
        print(num_cameras)
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            cam_id = camera_properties[0]
            print("camera id")
            print(cam_id)

            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = list(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            model_cameras[cam_id] = Camera(cam_id, model_name, width, height, params)
        assert len(model_cameras) == num_cameras
    return model_cameras


def read_images_text(path: str) -> List[Image]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    model_images: List[Image] = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                im_id = int(elements[0])
                qvec = list(map(float, elements[1:5]))
                tvec = list(map(float, elements[5:8]))
                cam_id = int(elements[8])
                image_name = elements[9]
                elements = fid.readline().split()
                point3d_ids = list(map(int, elements[2::3]))
                model_images.append(Image(im_id, qvec, tvec, cam_id, image_name, point3d_ids))
    return model_images


def read_images_binary(path: str) -> List[Image]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    model_images: List[Image] = []
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            im_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            cam_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points_2d = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points_2d, "ddq" * num_points_2d)
            point3d_ids = list(map(int, x_y_id_s[2::3]))
            model_images.append(Image(im_id, qvec, tvec, cam_id, image_name, point3d_ids))
    return model_images


def read_points_3d_text(path: str) -> Dict[int, Point3D]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    model_points3d: Dict[int, Point3D] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                point_id = int(elements[0])
                xyz = list(map(float, elements[1:4]))
                rgb = list(map(int, elements[4:7]))
                error = float(elements[7])
                image_ids = list(map(int, elements[8::2]))
                point2d_ids = list(map(int, elements[9::2]))
                model_points3d[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2d_ids)
    return model_points3d


def read_points3d_binary(path: str) -> Dict[int, Point3D]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    model_points3d: Dict[int, Point3D] = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = binary_point_line_properties[0]
            xyz = list(binary_point_line_properties[1:4])
            rgb = list(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elements = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = list(map(int, track_elements[0::2]))
            point2d_ids = list(map(int, track_elements[1::2]))
            model_points3d[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2d_ids)
    return model_points3d


def read_model(path: str, ext: str) -> Tuple[Dict[int, Camera], List[Image], Dict[int, Point3D]]:
    if ext == ".txt":
        model_cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        model_images = read_images_text(os.path.join(path, "images" + ext))
        model_points_3d = read_points_3d_text(os.path.join(path, "points3D") + ext)
    else:
        model_cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        model_images = read_images_binary(os.path.join(path, "images" + ext))
        model_points_3d = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return model_cameras, model_images, model_points_3d


def quaternion_to_rotation_matrix(qvec: List[float]) -> np.ndarray:
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def read_model_get_points(model_dir):
    '''
    读取colmap去畸变的稀疏建图结果
    Args:
        model_dir:
    Returns:所有图像的内参Dict，外参Dict，每个图像Dict关联的3D点

    '''
    cameras, images, points3d = read_model(model_dir, ".bin")
    num_images = len(images)

    param_type: Dict[str, List[str]] = {
        "SIMPLE_PINHOLE": ["f", "cx", "cy"],
        "PINHOLE": ["fx", "fy", "cx", "cy"],
        "SIMPLE_RADIAL": ["f", "cx", "cy", "k"],
        "SIMPLE_RADIAL_FISHEYE": ["f", "cx", "cy", "k"],
        "RADIAL": ["f", "cx", "cy", "k1", "k2"],
        "RADIAL_FISHEYE": ["f", "cx", "cy", "k1", "k2"],
        "OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
        "OPENCV_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
        "FULL_OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
        "FOV": ["fx", "fy", "cx", "cy", "omega"],
        "THIN_PRISM_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"]
    }

    # intrinsic
    intrinsic: Dict[int, np.ndarray] = {}
    for camera_id, cam in cameras.items():
        params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
        if "f" in param_type[cam.model]:
            params_dict["fx"] = params_dict["f"]
            params_dict["fy"] = params_dict["f"]
        i = np.array([
            [params_dict["fx"], 0, params_dict["cx"]],
            [0, params_dict["fy"], params_dict["cy"]],
            [0, 0, 1]
        ])
        intrinsic[camera_id] = i

    # extrinsic
    extrinsic: Dict[str,np.ndarray] = {}
    for i in range(num_images):
        e = np.zeros((4, 4))
        e[:3, :3] = quaternion_to_rotation_matrix(images[i].qvec)
        e[:3, 3] = images[i].tvec
        e[3, 3] = 1
        extrinsic[images[i].name] = e

    images_points: Dict[str,List[np.ndarray]] = {}
    for i in range(num_images):
        zs = []
        ex = extrinsic[images[i].name]
        for p3d_id in images[i].point3d_ids:
            if p3d_id == -1:
                continue
            transformed: np.ndarray = np.matmul(
                ex, [points3d[p3d_id].xyz[0], points3d[p3d_id].xyz[1], points3d[p3d_id].xyz[2], 1])
            zs.append(transformed[:3])
        images_points[images[i].name] = zs

    return intrinsic, extrinsic, images_points
