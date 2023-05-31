# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple
import albumentations as A
import cv2
import torch

from dataset.base_dataset import _BaseSODDataset
from dataset.transforms.resize import ms_resize, ss_resize
from dataset.transforms.rotate import UniRotate
from utils.builder import DATASETS
from utils.io.genaral import get_datasets_info_with_keys
from utils.io.image import read_color_array, read_gray_array
from PIL import Image

@DATASETS.register(name="MFFN_cod_te")
class MFFN_COD_TestDataset(_BaseSODDataset):
    def __init__(self, root: Tuple[str, dict], shape: Dict[str, int], interp_cfg: Dict = None):
        super().__init__(base_shape=shape, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=[root], extra_keys=["mask"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.image_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]

        image = read_color_array(image_path)
        image0 = cv2.flip(image, 0, dst=None)   # 作水平镜像翻转
        image1 = cv2.flip(image, -1, dst=None)  # 作垂直镜像翻转

        image = self.image_norm(image=image)["image"]
        image0 = self.image_norm(image=image0)["image"]
        image1 = self.image_norm(image=image1)["image"]

        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        image0 = ss_resize(image0, scale=1.0, base_h=base_h, base_w=base_w)
        image1 = ss_resize(image1, scale=1.0, base_h=base_h, base_w=base_w)
        images = ms_resize(image, scales=(2.0, 1.0, 1.8), base_h=base_h, base_w=base_w)
        image_c_1 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_o = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_c_2 = torch.from_numpy(images[2]).permute(2, 0, 1)
        image_a_1 = torch.from_numpy(image0).permute(2, 0, 1)
        image_a_2 = torch.from_numpy(image1).permute(2, 0, 1)

        return dict(
            data={
                "image_c1": image_c_1,
                "image_o": image_o,
                "image_c2": image_c_2,
                "image_a1": image_a_1,
                "image_a2": image_a_2,
            },
            info=dict(
                mask_path=mask_path,
            ),
        )

    def __len__(self):
        return len(self.total_image_paths)
@DATASETS.register(name="MFFN_cod_tr")
class MFFN_COD_TrainDataset(_BaseSODDataset):
    def __init__(
        self, root: List[Tuple[str, dict]], shape: Dict[str, int], extra_scales: List = None, interp_cfg: Dict = None
    ):
        super().__init__(base_shape=shape, extra_scales=extra_scales, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=root, extra_keys=["mask"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.joint_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                UniRotate(limit=10, interpolation=cv2.INTER_LINEAR, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
        self.reszie = A.Resize
    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]
        image = read_color_array(image_path)
        image0 = cv2.flip(image, 0, dst=None)  # 作水平镜像翻转
        image1 = cv2.flip(image, -1, dst=None)  # 作垂直镜像翻转
        mask = read_gray_array(mask_path, to_normalize=True, thr=0.5)
        transformed = self.joint_trans(image=image, mask=mask)
        transformed0 = self.joint_trans(image=image0, mask=mask)
        transformed1 = self.joint_trans(image=image1, mask=mask)
        image = transformed["image"]
        image0 = transformed0["image"]
        image1 = transformed1["image"]
        # image0 = cv2.flip(image, 0, dst=None)  # 作水平镜像翻转
        # image1 = cv2.flip(image, -1, dst=None)  # 作垂直镜像翻转
        mask = transformed["mask"]
        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(2.0, 1.0, 1.8), base_h=base_h, base_w=base_w)
        image0 = ss_resize(image0, scale=1.0, base_h=base_h, base_w=base_w)
        image1 = ss_resize(image1, scale=1.0, base_h=base_h, base_w=base_w)

        image_c_1 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_o = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_c_2 = torch.from_numpy(images[2]).permute(2, 0, 1)
        image_a_1 = torch.from_numpy(image0).permute(2, 0, 1)
        image_a_2 = torch.from_numpy(image1).permute(2, 0, 1)
        mask = ss_resize(mask, scale=1.0, base_h=base_h, base_w=base_w)
        mask_1_0 = torch.from_numpy(mask).unsqueeze(0)
        return dict(
            data={
                "image_c1": image_c_1,
                "image_o": image_o,
                "image_c2": image_c_2,
                "image_a1": image_a_1,
                "image_a2": image_a_2,
                "mask": mask_1_0,
            }
        )

    def __len__(self):
        return len(self.total_image_paths)
