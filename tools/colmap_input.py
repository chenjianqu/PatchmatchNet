import argparse
import cv2
import numpy as np
import os
import shutil
from typing import Dict, List, NamedTuple, Tuple
from colmap_utils import *
import PIL.Image as PImage


def calc_score(ind1: int, ind2: int) -> float:
    dist = np.linalg.norm(np.array(images[ind1].tvec) - np.array(images[ind2].tvec), ord=0)
    if dist > 100:
        return 0

    id_i = images[ind1].point3d_ids
    id_j = images[ind2].point3d_ids
    id_intersect = [it for it in id_i if it in id_j]
    cam_center_i = -np.matmul(extrinsic[ind1][:3, :3].transpose(), extrinsic[ind1][:3, 3:4])[:, 0]
    cam_center_j = -np.matmul(extrinsic[ind2][:3, :3].transpose(), extrinsic[ind2][:3, 3:4])[:, 0]
    view_score_ = 0.0
    for pid in id_intersect:
        if pid == -1:
            continue
        p = points3d[pid].xyz
        theta = (180 / np.pi) * np.arccos(
            np.dot(cam_center_i - p, cam_center_j - p) / np.linalg.norm(cam_center_i - p) / np.linalg.norm(
                cam_center_j - p))
        view_score_ += np.exp(-(theta - args.theta0) * (theta - args.theta0) / (
                2 * (args.sigma1 if theta <= args.theta0 else args.sigma2) ** 2))
    return view_score_


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


def read_crop_save(src, dst, crop_height):
    img = PImage.open(src)
    if crop_height > 0:
        width, height = img.size
        # 前两个坐标点是左上角坐标,后两个坐标点是右下角坐标(width在前， height在后)
        box = (0, 0, width, height - crop_height)
        img = img.crop(box)
    img.save(dst)


def parsing_args():
    parser = argparse.ArgumentParser(description="Convert colmap results into input for PatchmatchNet")

    parser.add_argument("--input_folder", type=str, help="Project input dir.")
    parser.add_argument("--output_folder", type=str, default="", help="Project output dir.")
    parser.add_argument("--num_src_images", type=int, default=-1, help="Related images")
    parser.add_argument("--crop_height", type=int, default=0, help="裁剪底部的高度（如果底部有发动机盖的话）")
    parser.add_argument("--theta0", type=float, default=5)
    parser.add_argument("--sigma1", type=float, default=1)
    parser.add_argument("--sigma2", type=float, default=10)
    parser.add_argument("--convert_format", action="store_true", default=False,
                        help="If set, convert image to jpg format.")

    args_raw = parser.parse_args()

    if not args_raw.output_folder:
        args_raw.output_folder = args_raw.input_folder

    if args_raw.input_folder is None or not os.path.isdir(args_raw.input_folder):
        raise Exception("Invalid input folder")

    if args_raw.output_folder is None or not os.path.isdir(args_raw.output_folder):
        raise Exception("Invalid output folder")

    return args_raw


if __name__ == "__main__":
    args = parsing_args()

    image_dir = os.path.join(args.input_folder, "images")
    mask_dir = os.path.join(args.input_folder, "un_mask")
    road_mask_dir = os.path.join(args.input_folder, "road_mask")
    model_dir = os.path.join(args.input_folder, "sparse")
    cam_dir = os.path.join(args.output_folder, "cams")
    renamed_dir = os.path.join(args.output_folder, "images")

    cameras, images, points3d = read_model(model_dir, ".bin")
    num_images = len(images)

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
    print("intrinsic[1]\n", intrinsic[1], end="\n\n")

    # extrinsic
    extrinsic: List[np.ndarray] = []
    for i in range(num_images):
        e = np.zeros((4, 4))
        e[:3, :3] = quaternion_to_rotation_matrix(images[i].qvec)
        e[:3, 3] = images[i].tvec
        e[3, 3] = 1
        extrinsic.append(e)
    print("extrinsic[0]\n", extrinsic[0], end="\n\n")

    # depth range and interval
    depth_ranges: List[Tuple[float, float]] = []
    for i in range(num_images):
        zs = []  # 所有特征点的深度
        for p3d_id in images[i].point3d_ids:
            if p3d_id == -1:
                continue
            transformed: np.ndarray = np.matmul(
                extrinsic[i], [points3d[p3d_id].xyz[0], points3d[p3d_id].xyz[1], points3d[p3d_id].xyz[2], 1])
            zs.append(transformed[2].item())
        zs_sorted = sorted(zs)
        # relaxed depth range
        # depth_min = zs_sorted[int(len(zs) * .01)]
        depth_min = 3.2
        depth_max = zs_sorted[int(len(zs) * .99)]

        depth_ranges.append((depth_min, depth_max))
    print("depth_ranges[0]\n", depth_ranges[0], end="\n\n")

    # view selection
    score = np.zeros((num_images, num_images))
    queue: List[Tuple[int, int]] = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            queue.append((i, j))
    cnt = 0
    q_size = len(queue)
    for i, j in queue:
        cnt += 1
        # s = calc_score(i, j)
        s = num_images - abs(i - j)  # 这里的假设是：图像已经按照时间进行排序
        score[i, j] = s
        score[j, i] = s
        print("[%d / %d] score:%f" % (cnt, q_size, s))

    if args.num_src_images < 0:
        args.num_src_images = num_images

    view_sel: List[List[Tuple[int, float]]] = []
    for i in range(num_images):
        sorted_score = np.argsort(score[i])[::-1]
        view_sel.append([(k, score[i, k]) for k in sorted_score[:args.num_src_images]])
    print("view_sel[0]\n", view_sel[0], end="\n\n")

    # write
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(renamed_dir, exist_ok=True)
    for i in range(num_images):
        with open(os.path.join(cam_dir, "%08d_cam.txt" % i), "w") as f:
            f.write("extrinsic\n")
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[i][j, k]) + " ")
                f.write("\n")
            f.write("\nintrinsic\n")
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[images[i].camera_id][j, k]) + " ")
                f.write("\n")
            f.write("\n%f %f \n" % (depth_ranges[i][0], depth_ranges[i][1]))

    kWinSize = 10
    with open(os.path.join(args.output_folder, "pair.txt"), "w") as f:
        f.write("%d\n" % len(images))
        for i, sorted_score in enumerate(view_sel):
            sorted_score = sorted_score[:kWinSize]
            f.write("%d\n%d " % (i, len(sorted_score)))
            for image_id, s in sorted_score:
                f.write("%d %f " % (image_id, s))
            f.write("\n")

    fout = open(os.path.join(args.output_folder, "name_map.txt"), "w")

    for i in range(num_images):
        src_path = os.path.join(image_dir, images[i].name)
        dst_path = os.path.join(renamed_dir, "%08d.jpg" % i)
        if args.convert_format:
            img = cv2.imread(src_path)  # 读进来后，将格式变为BGR
            if args.crop_height > 0:
                height, width, channel = img.shape
                img = img[0:height - args.crop_height, :]  # 裁剪坐标为[y0:y1, x0:x1]
            cv2.imwrite(dst_path, img)
        else:
            # shutil.copyfile(os.path.join(image_dir, images[i].name),
            #                os.path.join(renamed_dir, "%08d.jpg" % i))
            read_crop_save(src_path, dst_path, args.crop_height)

        # shutil.copyfile(os.path.join(mask_dir, images[i].name + ".png"),
        #                 os.path.join(mask_dir, "%08d.jpg" % i))
        read_crop_save(os.path.join(mask_dir, images[i].name + ".png"),
                       os.path.join(mask_dir, "%08d.jpg" % i), args.crop_height)

        time_stamp = images[i].name[:14]
        road_mask = cv2.imread(os.path.join(road_mask_dir, images[i].name + ".png"), 0)
        road_mask = road_mask[:1080, :3392]
        road_mask = cv2.resize(road_mask, dsize=(1696, 540))

        # 利用物体的mask 提高地面mask的精度
        obj_mask = cv2.imread(os.path.join(mask_dir, images[i].name + ".png"), 0)
        merge_mask = (road_mask > 0) & (obj_mask > 0)
        merge_mask = merge_mask.astype(np.int32)
        merge_mask = merge_mask * 255
        print(merge_mask.shape)

        height, width = merge_mask.shape
        merge_mask = merge_mask[0:height - args.crop_height, :]  # 裁剪坐标为[y0:y1, x0:x1]

        cv2.imwrite(os.path.join(road_mask_dir, "%08d.jpg" % i), merge_mask)
        fout.write("%i %s\n" % (i, images[i].name))

    fout.close()
