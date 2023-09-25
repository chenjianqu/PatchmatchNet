from typing import List

import numpy as np
import torch
import open3d as o3d


def visualize_depth(image, depth, K, prob):
    points = []

    height = depth.shape[0]
    width = depth.shape[1]

    # shape[W,H]
    xx, yy = np.meshgrid(np.arange(0, height), np.arange(0, width))
    xxx = xx.reshape([-1])  # [W*H]
    yyy = yy.reshape([-1])  # [W*H]
    ddd = depth.reshape([-1])

    x_pixel_d = np.stack([xxx, yyy, np.ones_like(xxx)], axis=0)  # [3,W*H]，引用图像的像素坐标
    x_pixel_d = x_pixel_d * ddd

    if prob.any():
        prob_t = prob.reshape([-1])
        prob_mask = prob_t > 0.8
        x_pixel_d = (x_pixel_d.transpose()[prob_mask]).transpose()

    x_cw = np.matmul(np.linalg.inv(K[0]), x_pixel_d)  # [3,W*H] = [3,3] * [3,W*H]
    x_cw = x_cw.transpose()  # [H*W,3]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='depth pcd')
    # vis.toggle_full_screen() #全屏
    # 设置
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # 背景
    opt.point_size = 1  # 点云大小

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(x_cw)
    vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window()

    # #位于原点的坐标轴
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])  # 坐标轴
    #
    #
    # #可视化
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(x_cw)
    # o3d.visualization.draw_geometries([point_cloud,mesh])


def vis_points(xyz: List[np.ndarray], colors: List[np.ndarray] = None):
    """

    Args:
        xyz: 多帧点云组成List,每个元素都是一个点云的 ndarray，which size is [N,3]
        colors: 多帧点云对应的颜色List，每个元素 [N,3]

    Returns:

    """
    assert xyz[0].shape[1] == 3

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='depth pcd')
    # vis.toggle_full_screen() #全屏
    # 设置
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # 背景
    opt.point_size = 1  # 点云大小

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    for i, points in enumerate(xyz):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            assert points.shape[0] == colors[i].shape[0]
            point_cloud.colors = o3d.utility.Vector3dVector(colors[i])

        vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window()


def write_ply(write_path: str, xyz: np.ndarray, colors: np.ndarray = None):
    """
    Args:
        write_path: 写入路径
        xyz: 点云ndarray，which size is [N,3]
        colors: 点云颜色[N,3]
    Returns:

    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz.tolist())
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(write_path, point_cloud)



