import argparse
import cv2
import numpy as np
import os
import sys
import time
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.parallel
from plyfile import PlyData, PlyElement
from typing import Tuple
from torch.utils.data import DataLoader

from datasets.data_io import read_cam_file, read_image, read_map, read_pair_file, save_image, save_map
from datasets.mvs_colmap import ColmapMVSDataset
from datasets.mvs_maxieye import MaxieyeMVSDataset
from models.module import generate_pointcloud
from models.net import PatchmatchNet
from tools.visualize_utils import vis_points
from utils import print_args, tensor2numpy, to_cuda
from PIL import Image


def save_depth(args):
    """Run MVS model to save depth maps"""
    if args.input_type == "params":
        print("Evaluating model with params from {}".format(args.checkpoint_path))
        model = PatchmatchNet(
            patchmatch_interval_scale=args.patchmatch_interval_scale,
            propagation_range=args.patchmatch_range,
            patchmatch_iteration=args.patchmatch_iteration,
            patchmatch_num_sample=args.patchmatch_num_sample,
            propagate_neighbors=args.propagate_neighbors,
            evaluate_neighbors=args.evaluate_neighbors
        )

        model = nn.DataParallel(model)
        state_dict = torch.load(args.checkpoint_path)["model"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Using scripted module from {}".format(args.checkpoint_path))
        model = torch.jit.load(args.checkpoint_path)
        model = nn.DataParallel(model)

    model.cuda()
    model.eval()

    test_datasets = ['20221027_6.58km_2022-12-04-11-38-23_part1', ]

    test_dataset = MaxieyeMVSDataset(
        data_root="/home/cjq/data/mvs/kaijin",
        lidar_data_root="/home/cjq/data/mvs/lidar",
        num_views=input_args.num_views,
        max_dim=input_args.image_max_dim,
        scan_list=test_datasets,
    )

    image_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    with torch.no_grad():
        for batch_idx, sample in enumerate(image_loader):
            start_time = time.time()
            sample_cuda = to_cuda(sample)
            depth, confidence, _ = model.forward(
                sample_cuda["images"],
                sample_cuda["intrinsics"],
                sample_cuda["extrinsics"],
                sample_cuda["depth_min"],
                sample_cuda["depth_max"],
            )
            depth = tensor2numpy(depth)
            confidence = tensor2numpy(confidence)
            del sample_cuda

            print(depth[0].shape)
            print(sample["images"][0][0].shape)
            print(sample["intrinsics"][0].shape)

            xyz, rgb = generate_pointcloud(depth[0][0], tensor2numpy(sample["images"][0][0]),
                                           tensor2numpy(sample["intrinsics"][0][0]))

            #xyz:[3,N]
            #rgb:[N,3]
            vis_points([xyz.transpose()],[rgb])


# project the reference point cloud into the source view, then project back
def reproject_with_depth(
        depth_ref: np.ndarray,
        intrinsics_ref: np.ndarray,
        extrinsics_ref: np.ndarray,
        depth_src: np.ndarray,
        intrinsics_src: np.ndarray,
        extrinsics_src: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the reference points to the source view, then project back to calculate the reprojection error

    Args:
        depth_ref: depths of points in the reference view, of shape (H, W)
        intrinsics_ref: camera intrinsic of the reference view, of shape (3, 3)
        extrinsics_ref: camera extrinsic of the reference view, of shape (4, 4)
        depth_src: depths of points in the source view, of shape (H, W)
        intrinsics_src: camera intrinsic of the source view, of shape (3, 3)
        extrinsics_src: camera extrinsic of the source view, of shape (4, 4)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            depth_reprojected: reprojected depths of points in the reference view, of shape (H, W)
            x_reprojected: reprojected x coordinates of points in the reference view, of shape (H, W)
            y_reprojected: reprojected y coordinates of points in the reference view, of shape (H, W)
    """
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    # step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # 参考相机坐标系3D点
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # 源相机坐标系3D点
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    k_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = k_xyz_src[:2] / k_xyz_src[2:3]  # 源像素坐标系

    # step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)  # 采样深度值

    # source 3D space，NOTE that we should use sampled source-view depth_here to project back
    # 将采样点变换到源相机坐标系
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # 将采样点投影回参考坐标系
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)  # 深度
    k_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = k_xyz_reprojected[:2] / k_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected


def check_geometric_consistency(
        depth_ref: np.ndarray,
        intrinsics_ref: np.ndarray,
        extrinsics_ref: np.ndarray,
        depth_src: np.ndarray,
        intrinsics_src: np.ndarray,
        extrinsics_src: np.ndarray,
        geo_pixel_thres: float,
        geo_depth_thres: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Check geometric consistency and return valid points

    Args:
        depth_ref: depths of points in the reference view, of shape (H, W)
        intrinsics_ref: camera intrinsic of the reference view, of shape (3, 3)
        extrinsics_ref: camera extrinsic of the reference view, of shape (4, 4)
        depth_src: depths of points in the source view, of shape (H, W)
        intrinsics_src: camera intrinsic of the source view, of shape (3, 3)
        extrinsics_src: camera extrinsic of the source view, of shape (4, 4)
        geo_pixel_thres: geometric pixel threshold
        geo_depth_thres: geometric depth threshold

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            mask: mask for points with geometric consistency, of shape (H, W)
            depth_reprojected: reprojected depths of points in the reference view, of shape (H, W)
    """
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src)

    # check |p_reproject - p_1| < 1
    # 源深度对应的像素点，和参考深度对应的像素点，的距离
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproject - d_1| / d_1 < 0.01
    # 源深度和参考深度的距离
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected


def filter_depth(args, scan: str = ""):
    # the pair file
    pair_file = os.path.join(args.input_folder, scan, "pair.txt")
    # for the final point cloud
    vertices = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the reference image
        ref_img, original_h, original_w = read_image(
            os.path.join(args.input_folder, scan, "images/{:0>8}.jpg".format(ref_view)), args.image_max_dim)
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_cam_file(
            os.path.join(args.input_folder, scan, "cams/{:0>8}_cam.txt".format(ref_view)))[0:2]
        ref_intrinsics[0] *= ref_img.shape[1] / original_w
        ref_intrinsics[1] *= ref_img.shape[0] / original_h

        # load the estimated depth of the reference view
        ref_depth_est = read_map(
            os.path.join(args.output_folder, scan, "depth_est/{:0>8}{}".format(ref_view, args.file_format))).squeeze(2)
        # load the photometric mask of the reference view
        confidence = read_map(
            os.path.join(args.output_folder, scan, "confidence/{:0>8}{}".format(ref_view, args.file_format)))

        # 光度mask
        photo_mask = (confidence > args.photo_thres).squeeze(2)

        # 语义mask,读取并转换为二值图像
        semantic_mask = None
        if args.use_road_mask:
            mask_image_raw = Image.open(
                os.path.join(args.input_folder, scan, "road_mask/{:0>8}.jpg".format(ref_view))).convert('L')
            semantic_mask = np.array(mask_image_raw, dtype=np.int32)
            semantic_mask = semantic_mask > 0

        all_src_view_depth_estimates = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_image, original_h, original_w = read_image(
                os.path.join(args.input_folder, scan, "images/{:0>8}.jpg".format(src_view)), args.image_max_dim)
            src_intrinsics, src_extrinsics = read_cam_file(
                os.path.join(args.input_folder, scan, "cams/{:0>8}_cam.txt".format(src_view)))[0:2]
            src_intrinsics[0] *= src_image.shape[1] / original_w
            src_intrinsics[1] *= src_image.shape[0] / original_h

            # the estimated depth of the source view
            src_depth_est = read_map(
                os.path.join(args.output_folder, scan, "depth_est/{:0>8}{}".format(src_view, args.file_format)))

            geo_mask, depth_reprojected = check_geometric_consistency(
                ref_depth_est,
                ref_intrinsics,
                ref_extrinsics,
                src_depth_est,
                src_intrinsics,
                src_extrinsics,
                args.geo_pixel_thres,
                args.geo_depth_thres
            )
            geo_mask_sum += geo_mask.astype(np.int32)
            all_src_view_depth_estimates.append(depth_reprojected)

        depth_est_averaged = (sum(all_src_view_depth_estimates) + ref_depth_est) / (geo_mask_sum + 1)

        geo_mask = geo_mask_sum >= args.geo_mask_thres
        final_mask = np.logical_and(photo_mask, geo_mask)
        if args.use_road_mask:
            final_mask = np.logical_and(final_mask, semantic_mask)

        # final_mask = semantic_mask

        os.makedirs(os.path.join(args.output_folder, scan, "mask"), exist_ok=True)
        save_image(os.path.join(args.output_folder, scan, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_image(os.path.join(args.output_folder, scan, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_image(os.path.join(args.output_folder, scan, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>3}, geo_mask:{:3f}, photo_mask:{:3f}, final_mask: {:3f}".format(
            os.path.join(args.input_folder, scan), ref_view, geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

        if args.display:
            cv2.imshow("ref_img", ref_img[:, :, ::-1])
            cv2.imshow("ref_depth", ref_depth_est)
            cv2.imshow("ref_depth * photo_mask", ref_depth_est * photo_mask.astype(np.float32))
            cv2.imshow("ref_depth * geo_mask", ref_depth_est * geo_mask.astype(np.float32))
            cv2.imshow("ref_depth * mask", ref_depth_est * final_mask.astype(np.float32))
            cv2.waitKey(1)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        x, y, depth = x[final_mask], y[final_mask], depth_est_averaged[final_mask]

        color = ref_img[final_mask]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics), np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics), np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertices.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertices = np.concatenate(vertices, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertices = np.array([tuple(v) for v in vertices], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")])

    vertex_all = np.empty(len(vertices), vertices.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertices.dtype.names:
        vertex_all[prop] = vertices[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, "vertex")
    ply_filename = os.path.join(args.output_folder, scan, "fused.ply")
    PlyData([el]).write(ply_filename)
    print("saving the final model to", ply_filename)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Predict depth, filter, and fuse")

    # High level input/output options
    parser.add_argument("--input_folder", default="/home/cjq/data/mvs", type=str, help="input data path")
    parser.add_argument("--mask_folder", type=str, default='', help="input mask path")
    parser.add_argument("--output_folder", type=str, default="/home/cjq/data/mvs", help="output path")
    parser.add_argument("--checkpoint_path", default="/home/cjq/PycharmProjects/MVS/PatchmatchNet/checkpoints/params_000007.ckpt",
                        type=str, help="load a specific checkpoint for parameters of model")
    parser.add_argument("--file_format", type=str, default=".pfm", help="File format for depth maps",
                        choices=[".bin", ".pfm"])
    parser.add_argument("--input_type", type=str, default="params", help="Input type of checkpoint",
                        choices=["params", "module"])
    parser.add_argument("--output_type", type=str, default="both", help="Type of outputs to produce",
                        choices=["depth", "fusion", "both"])
    parser.add_argument("--colmap_dense_folder", type=str, default='', help="input colmap_dense_folder path")

    # Dataset loading options
    parser.add_argument("--num_views", type=int, default=5,
                        help="number of source views for each patch-match problem")
    parser.add_argument("--image_max_dim", type=int, default=960, help="max image dimension")
    parser.add_argument("--scan_list", type=str, default="",
                        help="Optional scan list text file to identify input folders")
    parser.add_argument("--num_light_idx", type=int, default=-1, help="Number of light indexes in source images")
    parser.add_argument("--batch_size", type=int, default=1, help="evaluation batch size")

    # PatchMatchNet module options (only used when not loading from file)
    # parser.add_argument("--patchmatch_interval_scale", nargs="+", type=float, default=[0.005, 0.0125, 0.025],
    #                    help="normalized interval in inverse depth range to generate samples in local perturbation")
    parser.add_argument("--patchmatch_interval_scale", nargs="+", type=float, default=[0.005, 0.0125, 0.025],
                        help="normalized interval in inverse depth range to generate samples in local perturbation")
    parser.add_argument("--patchmatch_range", nargs="+", type=int, default=[6, 4, 2],
                        help="fixed offset of sampling points for propagation of patch match on stages 1,2,3")
    parser.add_argument("--patchmatch_iteration", nargs="+", type=int, default=[1, 2, 2],
                        help="num of iteration of patch match on stages 1,2,3")
    parser.add_argument("--patchmatch_num_sample", nargs="+", type=int, default=[8, 8, 16],
                        help="num of generated samples in local perturbation on stages 1,2,3")
    parser.add_argument("--propagate_neighbors", nargs="+", type=int, default=[0, 8, 16],
                        help="num of neighbors for adaptive propagation on stages 1,2,3")
    parser.add_argument("--evaluate_neighbors", nargs="+", type=int, default=[9, 9, 9],
                        help="num of neighbors for adaptive matching cost aggregation of adaptive evaluation on stages 1,2,3")

    # Stereo fusion options
    parser.add_argument("--display", action="store_true", default=False, help="display depth images and masks")
    parser.add_argument("--use_road_mask", action="store_true", default=False, help="display depth images and masks")

    parser.add_argument("--geo_pixel_thres", type=float, default=1.0,
                        help="pixel threshold for geometric consistency filtering")
    parser.add_argument("--geo_depth_thres", type=float, default=0.01,
                        help="depth threshold for geometric consistency filtering")
    parser.add_argument("--geo_mask_thres", type=int, default=5, help="threshold for geometric consistency filtering")
    parser.add_argument("--photo_thres", type=float, default=0.5,
                        help="threshold for photometric consistency filtering")

    # parse arguments and check
    input_args = parser.parse_args()
    print("argv: ", sys.argv[1:])
    print_args(input_args)

    if input_args.input_folder is None or not os.path.isdir(input_args.input_folder):
        raise Exception("Invalid input folder: {}".format(input_args.input_folder))

    if input_args.checkpoint_path is None or not os.path.isfile(input_args.checkpoint_path):
        raise Exception("Invalid checkpoint file: {}".format(input_args.checkpoint_path))

    if not input_args.output_folder:
        input_args.output_folder = input_args.input_folder

    # Create output folder if it does not exist
    os.makedirs(input_args.output_folder, exist_ok=True)

    save_depth(input_args)
    # We can free all the GPU memory here since we don't need it for the fusion part
    torch.cuda.empty_cache()
