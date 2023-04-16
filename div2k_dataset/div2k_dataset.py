import os
import glob
import torch
import random
import albumentations as A
from torch.utils.data import Dataset

import bsrgan_degradation_model.utils.utils_image as util
import bsrgan_degradation_model.utils.utils_blindsr as blindsr
from matlab_imresize.imresize import imresize as matlab_imresize


class Div2KDataset(Dataset):
    def __init__(self, path, transform=None, img_ext="png", scale_factor=4, shuffle_prob=0.1, use_sharp=True,
                 lq_patch_size=72, apply_bicubic=0.2, use_center_crop=False):
        self.transform = transform
        self.scale_factor = scale_factor
        self.shuffle_prob = shuffle_prob
        self.use_sharp = use_sharp
        self.lq_patch_size = lq_patch_size
        self.apply_bicubic = apply_bicubic
        self.use_center_crop = use_center_crop
        self.image_paths = glob.glob(os.path.join(path, f"**/*.{img_ext}"), recursive=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # get the image
        img = util.imread_uint(image_path)  # read RGB image; shape (H, W, C); range [0, 255]

        # optionally apply augmentations on the image (e.g., rotation, horizontal flip, etc)
        if self.transform:
            img = self.transform(image=img)["image"]

        # Either apply MATLAB bicubic down-sampling or BSRGAN+ complex degradation
        if random.random() < self.apply_bicubic:
            size = self.lq_patch_size * self.scale_factor
            crop_fn = A.CenterCrop if self.use_center_crop else A.RandomCrop
            img_hq = crop_fn(height=size, width=size)(image=img)["image"]
            img_lq = matlab_imresize(img_hq, output_shape=(self.lq_patch_size, self.lq_patch_size))
            img_lq, img_hq = map(lambda image: util.uint2single(image), (img_lq, img_hq))
        else:
            img = util.uint2single(img)  # convert to range [0, 1] (expected by "blindsr.degradation_bsrgan_plus")
            img_lq, img_hq = blindsr.degradation_bsrgan_plus(img,
                                                             sf=self.scale_factor,
                                                             shuffle_prob=self.shuffle_prob,
                                                             use_sharp=self.use_sharp,
                                                             lq_patchsize=self.lq_patch_size,
                                                             use_center_crop=self.use_center_crop)
            # img_lq shape (PATCH_SIZE, PATCH_SIZE, C)
            # img_lq shape (PATCH_SIZE * SCALE_FACTOR, PATCH_SIZE * SCALE_FACTOR, C)
            # range for both [0, 1]

        # convert from numpy arrays to PyTorch tensors
        img_lq, img_hq = map(lambda arr: torch.from_numpy(arr.transpose(2, 0, 1)), (img_lq, img_hq))
        return {"gt": img_hq, "lq": img_lq}
