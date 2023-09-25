"""
Implementation of Pytorch layer primitives, such as Conv+BN+ReLU, differentiable warping layers,
and depth regression based upon expectation of an input probability distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBnReLU(nn.Module):
    """Implements 2d Convolution + batch normalization + ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            pad: int = 1,
            dilation: int = 1,
    ) -> None:
        """initialization method for convolution2D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU3D(nn.Module):
    """Implements of 3d convolution + batch normalization + ReLU."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            pad: int = 1,
            dilation: int = 1,
    ) -> None:
        """initialization method for convolution3D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU1D(nn.Module):
    """Implements 1d Convolution + batch normalization + ReLU."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            pad: int = 1,
            dilation: int = 1,
    ) -> None:
        """initialization method for convolution1D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    """Implements of 2d convolution + batch normalization."""

    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pad: int = 1
    ) -> None:
        """initialization method for convolution2D + batch normalization + ReLU module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
        """
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return self.bn(self.conv(x))


def differentiable_warping(
        src_fea: torch.Tensor, src_proj: torch.Tensor, ref_proj: torch.Tensor, depth_samples: torch.Tensor
):
    """Differentiable homography-based warping, implemented in Pytorch.

    Args:
        src_fea: [B, C, Hin, Win] source features, for each source view in batch
        src_proj: [B, 4, 4] source camera projection matrix, for each source view in batch
        ref_proj: [B, 4, 4] reference camera projection matrix, for each ref view in batch
        depth_samples: [B, Ndepth, Hout, Wout] virtual depth layers
    Returns:
        warped_src_fea: [B, C, Ndepth, Hout, Wout] features on depths after perspective transformation
    """

    batch, num_depth, height, width = depth_samples.shape
    channels = src_fea.shape[1]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                torch.arange(0, width, dtype=torch.float32, device=src_fea.device),
            ]
        )
        y, x = y.contiguous().view(height * width), x.contiguous().view(height * width)
        xyz = torch.unsqueeze(torch.stack((x, y, torch.ones_like(x))), 0).repeat(batch, 1, 1)  # [B, 3, H*W]

        rot_depth_xyz = torch.matmul(rot, xyz).unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(
            batch, 1, num_depth, height * width
        )  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # avoid negative depth（由于有些区域视野并不重叠，因此得到的深度可能是负的）
        negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
        proj_xyz[:, 0:1][negative_depth_mask] = float(width)
        proj_xyz[:, 1:2][negative_depth_mask] = float(height)
        proj_xyz[:, 2:3][negative_depth_mask] = 1.0
        # 得到参考视图的每个深度在源视图下的像素坐标
        grid = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = grid[:, 0, :, :] / ((width - 1) / 2) - 1  # [B, Ndepth, H*W]
        proj_y_normalized = grid[:, 1, :, :] / ((height - 1) / 2) - 1
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]

    # 采样源视图上的特征
    return F.grid_sample(
        src_fea,
        grid.view(batch, num_depth * height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch, channels, num_depth, height, width)


def depth_regression(p: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
    """Implements per-pixel depth regression based upon a probability distribution per-pixel.

    The regressed depth value D(p) at pixel p is found as the expectation w.r.t. P of the hypotheses.

    Args:
        p: probability volume [B, D, H, W]
        depth_values: discrete depth values [B, D]
    Returns:
        result depth: expected value, soft argmin [B, 1, H, W]
    """

    return torch.sum(p * depth_values.view(depth_values.shape[0], 1, 1), dim=1).unsqueeze(1)


def is_empty(x: torch.Tensor) -> bool:
    return x.numel() == 0


def ground_plane_init_depth(width, height, K):
    '''
    地面深度假设
    @param width: 图像宽度
    @param height: 图像高度
    @param K: 内参矩阵 [3,3]
    @return: 地面假设的深度 [height,width]
    '''
    T_vc = np.array([[1.65671625e-02, -3.32327391e-02, 9.99310319e-01, 2.17537799e+00],
                     [9.99862347e-01, 3.52411779e-04, 1.65880340e-02, 2.80948292e-02],
                     [-9.03434534e-04, -9.99447578e-01, -3.32223260e-02, 1.33101139e+00],
                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], np.float32)
    T_cv = np.linalg.inv(T_vc)

    K_inv = np.linalg.inv(K)

    # 生成相机坐标系下的归一化坐标
    x_cam, y_cam = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_cam, y_cam = x_cam.reshape(-1), y_cam.reshape(-1)
    p_cam_pixel = np.vstack([x_cam, y_cam, np.ones([1, width * height], dtype=np.float32)])  # [3, H*W]
    p_cam_norm = np.matmul(K_inv, p_cam_pixel)  # [3, H*W]
    p_cam_norm_h = np.vstack([p_cam_norm, np.ones_like(x_cam)])  # [4, H*W]
    # 变换到car坐标系
    p_car_norm = np.matmul(T_vc, p_cam_norm_h)[:3, :]  # [3,H*W]

    # 相机光心
    O_car_cam = (T_vc[:3, 3]).reshape([3, 1])  # [3,1]
    # 计算射线的方向向量
    v_car = p_car_norm - O_car_cam
    v_car = v_car / np.linalg.norm(v_car, ord=2, axis=0)  # 归一化

    # 平面法向量
    N_plane = np.array([0, 0, 1], np.float32).reshape([3, 1])
    # 计算k
    temp_1 = np.dot(N_plane.transpose(), -O_car_cam)
    k = temp_1 / np.matmul(v_car.transpose(), N_plane)  # [H*W,1]
    # 计算射线和平面的交点
    p_car_inter = O_car_cam + np.squeeze(k, axis=1) * v_car  # [3,H*W]
    # 将点变换到相机坐标系
    p_cam_inter = np.matmul(T_cv, np.vstack([p_car_inter, np.ones([1, width * height])]))[:3, :]  # [4,H*W]

    # 获得深度
    depth = p_cam_inter[2]  # [1,H*W]

    return depth.reshape([height, width]).astype(np.float32)


def construct_prior_depth(width, height, K, points):
    '''
    根据已知的3D点，构造深度先验
    Args:
        width: 图像宽
        height: 图像高
        K: 内参矩阵[B,3,3]
        points: 3D点[1,3,N]

    Returns: [height,width]的深度图
    '''
    xyd = torch.matmul(K, points)  # [B,3,N]
    xyd = xyd[0]  # [3,N]
    d = xyd[2]  # [N]
    xy = (xyd / d)[:2, :]  # [2,N]
    xy = xy.int()

    # 删掉一些点
    x_mask_0 = (xy[0, :] > 0)
    x_mask_1 = (xy[0, :] < width)
    y_mask_0 = (xy[1, :] > 0)
    y_mask_1 = (xy[1, :] < height)
    d_mask_0 = d > 0
    d_mask_1 = d < 150
    final_mask = x_mask_0 & x_mask_1 & y_mask_0 & y_mask_1 & d_mask_0 & d_mask_1

    d = d[final_mask]
    xy = xy[:, final_mask]

    depth_img = torch.zeros([height, width], dtype=torch.float32, device=points.device)
    depth_img[xy[1], xy[0]] = d

    return depth_img


def plane_fit(points):
    '''
    直接法平面拟合，https://zhuanlan.zhihu.com/p/390294059
    Args:
        points: 3D点，[B,3,N]
    Returns:
        平面系数 [4], A,B,C,D
    '''
    # ------------------------构建系数矩阵-----------------------------
    N = points.shape[2]
    sum_X = torch.sum(points[:, 0, :])
    sum_Y = torch.sum(points[:, 1, :])
    sum_Z = torch.sum(points[:, 2, :])
    sum_XY = torch.sum(points[:, 0, :] * points[:, 1, :])
    sum_X2 = torch.sum(points[:, 0, :] * points[:, 0, :])
    sum_Y2 = torch.sum(points[:, 1, :] * points[:, 1, :])
    sum_XZ = torch.sum(points[:, 0, :] * points[:, 2, :])
    sum_YZ = torch.sum(points[:, 1, :] * points[:, 2, :])

    A = torch.Tensor([[sum_X2, sum_XY, sum_X],
                      [sum_XY, sum_Y2, sum_Y],
                      [sum_X, sum_Y, N]])
    B = torch.Tensor([sum_XZ, sum_YZ, sum_Z])
    X = torch.linalg.solve(A, B)
    paras = torch.tensor([X[0], X[1], -1, X[2]], device=points.device)
    return paras


def solve_intersection(plane_paras, O, dirs):
    '''
    根据空间平面方程，和空间点O，求从O发射的射线与平面的交点坐标
    Args:
        plane_paras:平面的参数,shape(4)
        O:空间起点，shape[B,3]
        dirs:射线的方向 ,shape[B,3,N]
    Returns: 与平面的交点 [3,N]
    '''
    batch = dirs.shape[0]
    O = O.view([batch, 3, 1])  # [B,3,1]
    # 平面法向量
    normal = plane_paras[:3].view([batch, 3, 1])  # [B,3,1]
    # 平面上的某一点
    c = torch.tensor([0, 0, - plane_paras[3] / plane_paras[2]], dtype=torch.float32,
                     device=dirs.device).reshape([batch, 3, 1])  # [B,3,1]
    # 计算k
    temp_1 = torch.matmul(normal.transpose(dim0=1, dim1=2), (c - O))  # [B,1,1]
    k = temp_1 / torch.matmul(dirs.transpose(dim0=1, dim1=2), normal)  # [B,N,1]
    # 计算射线和平面的交点
    intersection = O + k.squeeze(2) * dirs  # [B,3,N]
    return intersection


def grid_plane_fit(depth, mask, K):
    '''
    在mask区域内，划分网格，并对每个网格拟合平面，将拟合得到的平面的深度设置为新的深度
    Args:
        depth:[B,1,H,W] 已有的深度图
        mask:[B,H,W] mask
        K:(B,3,3), 内参矩阵
    Returns:
    '''
    batch_size, _, height, width = depth.shape
    device = depth.device
    grid_size = int(height / 10)  # 用于估计平面的网格大小

    K_inv = torch.linalg.inv(K)  # [B,3,3]

    for start_y in range(0, height - grid_size, grid_size):
        for start_x in range(0, width - grid_size, grid_size):
            end_y = start_y + grid_size
            end_x = start_x + grid_size
            mask_roi = mask[:, start_y:end_y, start_x:end_x]  # [B,S,S]
            if mask_roi.sum() < 5:
                continue
            mask_roi = mask_roi.reshape([batch_size, -1])

            y_grid, x_grid = torch.meshgrid(
                [
                    torch.arange(start_y, end_y, dtype=torch.float32, device=device),
                    torch.arange(start_x, end_x, dtype=torch.float32, device=device),
                ]
            )
            y_grid, x_grid = y_grid.contiguous().view(grid_size * grid_size), x_grid.contiguous().view(
                grid_size * grid_size)
            xy = torch.stack((x_grid, y_grid, torch.ones_like(x_grid)))  # [3, S*S]
            xy = torch.unsqueeze(xy, 0).repeat(batch_size, 1, 1)  # [B, 3, S*S]
            xyd = xy * depth[:, :, start_y:end_y, start_x:end_x].reshape([batch_size, -1])  # [B, 3, S*S]
            xyz = torch.matmul(K_inv, xyd)  # [B, 3, S*S]

            # 平面拟合
            xyz_masked = xyz[:, :, mask_roi[0]]  # [B, 3, N]
            planes = plane_fit(xyz_masked)  # [4]

            # 求射线交点
            O = torch.zeros([batch_size, 3, 1], dtype=torch.float32, device=device)  # [B,3,1]
            dirs = xyz / xyz[:, 2]  # [B,3,N]
            points_inter = solve_intersection(planes, O, dirs)  # [B,3,N]

            # 投影交点，得到新的深度
            xyd = torch.matmul(K, points_inter)  # [B,3,N]
            d_new = xyd[:, 2]  # [B,N]

            # 将新的深度应用到输入的深度图中
            d = d_new.reshape(batch_size, grid_size, grid_size)  # [B,S,S]
            mask_depth = mask_roi.reshape(batch_size, grid_size, grid_size)
            (depth[:, :, start_y:end_y, start_x:end_x])[mask_depth.unsqueeze(1)] = d[mask_depth]


def generate_pointcloud(depth_image: np.ndarray, rgb_image: np.ndarray, K: np.ndarray):
    """
    从深度图中生成点云
    Args:
        K: [3,3]相机内参
        depth_image:深度图 [H,W]
        rgb_image:彩色图 [3,H,W] 或 [H,W,3]
    Returns: 点云数组xyz [3,H*W]，颜色数组rgb [H*W,3]
    """
    H, W = depth_image.shape
    K_inv = np.linalg.inv(K)

    zz = depth_image.reshape([-1])  # N

    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, H))
    xx, yy = xx.reshape(-1), yy.reshape(-1)
    xyd_pixel = np.vstack([xx, yy, np.ones([1, H * W], dtype=np.float32)]) * zz  # [3, H*W]
    xyz = np.matmul(K_inv, xyd_pixel)  # [3, H*W]

    if rgb_image.shape[0] == 3:
        rgb_image = rgb_image.transpose(1, 2, 0)

    # 此时 rgb_image：[H,W,3]
    rgb = rgb_image.reshape([-1, 3])

    return xyz, rgb
