import numpy as np
import os
import random

from PIL import Image

from datasets.data_io import read_cam_file, read_image, read_pair_file, read_binary_mask
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from tools.colmap_utils import read_model_get_points


class ColmapMVSDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            num_views: int = 10,
            max_dim: int = -1,
            scan_list: str = '',
            num_light_idx: int = -1,
            cam_folder: str = "cams",
            pair_path: str = "pair.txt",
            image_folder: str = "images",
            depth_folder: str = "depth_gt",
            image_extension: str = ".jpg",
            robust_train: bool = False,
            mask_folder: str = '',
            colmap_folder: str = ''
    ) -> None:
        super(ColmapMVSDataset, self).__init__()

        self.data_path = data_path
        self.num_views = num_views
        self.max_dim = max_dim
        self.robust_train = robust_train
        self.cam_folder = cam_folder
        self.depth_folder = depth_folder
        self.image_folder = image_folder
        self.image_extension = image_extension
        self.metas: List[Tuple[str, str, int, List[int]]] = []
        self.mask_folder = mask_folder
        self.colmap_folder = colmap_folder

        self.colmap_intrinsic: Dict[int, np.ndarray] = {}
        self.colmap_extrinsic: Dict[str, np.ndarray] = {}
        self.colmap_images_points: Dict[str, List[np.ndarray]] = {}

        if os.path.isfile(scan_list):
            with open(scan_list) as f:
                scans = [line.rstrip() for line in f.readlines()]
        else:
            scans = ['']

        if num_light_idx > 0:
            light_indexes = [str(idx) for idx in range(num_light_idx)]
        else:
            light_indexes = ['']

        for scan in scans:
            pair_data = read_pair_file(os.path.join(self.data_path, scan, pair_path))
            for light_idx in light_indexes:
                self.metas += [(scan, light_idx, ref, src) for ref, src in pair_data]

        if self.colmap_folder != '':
            self.name_map: Dict[str, str] = {}
            with open(os.path.join(self.colmap_folder, 'name_map.txt')) as f:
                for line in f.readlines():
                    idx, name = line.rstrip().split()
                    self.name_map[idx] = name

            colmap_read_path = os.path.join(self.colmap_folder, 'sparse')
            self.colmap_intrinsic, self.colmap_extrinsic, self.colmap_images_points = read_model_get_points(
                colmap_read_path)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scan, light_idx, ref_view, src_views = self.metas[idx]
        # use only the reference view and first num_views source views
        num_src_views = min(len(src_views), self.num_views)
        if self.robust_train:
            index = random.sample(range(len(src_views)), num_src_views)
            view_ids = [ref_view] + [src_views[i] for i in index]
        else:
            view_ids = [ref_view] + src_views[:num_src_views]

        images = []
        intrinsics = []
        extrinsics = []
        depth_min: float = -1.0
        depth_max: float = -1.0
        depth_gt = np.empty(0)
        mask = None  # 道路掩码区域
        prior_points = np.empty(0)  # 参考帧中的已知特征点

        for view_index, view_id in enumerate(view_ids):
            img_filename = os.path.join(
                self.data_path, scan, self.image_folder, light_idx, "{:0>8}{}".format(view_id, self.image_extension))

            image, original_h, original_w = read_image(img_filename, self.max_dim)
            # image, original_h, original_w = read_image_set_mask(img_filename, self.max_dim)
            curr_img_h = image.shape[0]
            curr_img_w = image.shape[1]
            images.append(image.transpose([2, 0, 1]))

            cam_filename = os.path.join(self.data_path, scan, self.cam_folder, "{:0>8}_cam.txt".format(view_id))
            intrinsic, extrinsic, depth_params = read_cam_file(cam_filename)

            intrinsic[0] *= image.shape[1] / original_w
            intrinsic[1] *= image.shape[0] / original_h
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            if view_index == 0:  # reference view
                depth_min = depth_params[0]
                depth_max = depth_params[1]
                depth_gt_filename = os.path.join(self.data_path, scan, self.depth_folder, "{:0>8}.pfm".format(view_id))

                # if os.path.isfile(depth_gt_filename):
                #     # Using `copy()` here to avoid the negative stride resulting from the transpose
                #     depth_gt = read_map(depth_gt_filename, self.max_dim).transpose([2, 0, 1]).copy()
                #     # Create mask from GT depth map
                #     mask = depth_gt >= depth_min

                if self.mask_folder != '':
                    mask_filename = os.path.join(
                        self.data_path, scan, self.mask_folder, light_idx,
                        "{:0>8}{}".format(view_id, self.image_extension))
                    mask = read_binary_mask(mask_filename, img_size=(curr_img_w, curr_img_h))
                    mask = mask.astype(np.float32)

                if self.colmap_images_points:
                    name = self.name_map[str(view_id)]
                    prior_points = np.array(self.colmap_images_points[name]).astype(np.float32)  # 得到[N,3]
                    prior_points = prior_points.transpose()  # [3,N]

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)

        return {
            "images": images,  # List[Tensor]: [N][3,Hi,Wi], N is number of images
            "intrinsics": intrinsics,  # Tensor: [N,3,3]
            "extrinsics": extrinsics,  # Tensor: [N,4,4]
            "depth_min": depth_min,  # Tensor: [1]
            "depth_max": depth_max,  # Tensor: [1]
            "depth_gt": depth_gt,  # Tensor: [1,H0,W0] if exists
            "mask": mask,  # Tensor: [1,H0,W0] if exists
            "filename": os.path.join(scan, "{}", "{:0>8}".format(view_ids[0]) + "{}"),
            "prior_points": prior_points,  # [3,N]
        }
