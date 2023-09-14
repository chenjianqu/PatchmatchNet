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
        # avoid negative depth
        negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
        proj_xyz[:, 0:1][negative_depth_mask] = float(width)
        proj_xyz[:, 1:2][negative_depth_mask] = float(height)
        proj_xyz[:, 2:3][negative_depth_mask] = 1.0
        grid = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = grid[:, 0, :, :] / ((width - 1) / 2) - 1  # [B, Ndepth, H*W]
        proj_y_normalized = grid[:, 1, :, :] / ((height - 1) / 2) - 1
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]

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

    A = np.array([[sum_X2, sum_XY, sum_X],
                  [sum_XY, sum_Y2, sum_Y],
                  [sum_X, sum_Y, N]], dtype=np.float32)
    A = torch.from_numpy(A)

    B = np.array([[sum_XZ, sum_YZ, sum_Z]], dtype=np.float32)
    B = torch.from_numpy(B)

    X = torch.linalg.solve(A, B)

    return X



def solve_intersection(plane_paras,O,dirs):
    '''
    根据空间平面方程，和空间点O，求从O发射的射线与平面的交点坐标
    Args:
        plane_paras:平面的参数,shape(4)
        O:空间起点，shape[3]
        dirs:射线的方向 ,shape[3,N]
    Returns: 与平面的交点 [3,N]
    '''










