import os
import glob
import torch
from torch.utils.data import Dataset

import bsrgan_degradation.utils_image as util
import bsrgan_degradation.utils_blindsr as blindsr


class Div2KDataset(Dataset):
    def __init__(self, path, transform=None, img_ext="png", scale_factor=4, shuffle_prob=0.1, use_sharp=True,
                 patch_size=72):
        self.transform = transform
        self.scale_factor = scale_factor
        self.shuffle_prob = shuffle_prob
        self.use_sharp = use_sharp
        self.patch_size = patch_size
        self.image_paths = glob.glob(os.path.join(path, f"**/*.{img_ext}"), recursive=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # get the image
        img = util.imread_uint(image_path)  # read RGB image; shape (H, W, C); range [0, 255]
        img = util.uint2single(img)  # convert to range [0, 1] (expected by "blindsr.degradation_bsrgan_plus")

        # optionally apply augmentations on the image (e.g., rotation, horizontal flip, etc)
        if self.transform:
            img = self.transform(image=img)["image"]

        # img_lq shape (PATCH_SIZE, PATCH_SIZE, C)
        # img_lq shape (PATCH_SIZE * SCALE_FACTOR, PATCH_SIZE * SCALE_FACTOR, C)
        # range for both [0, 1]
        img_lq, img_hq = blindsr.degradation_bsrgan_plus(img,
                                                         sf=self.scale_factor,
                                                         shuffle_prob=self.shuffle_prob,
                                                         use_sharp=self.use_sharp,
                                                         lq_patchsize=self.patch_size)
        # convert from numpy arrays to PyTorch tensors
        img_lq, img_hq = map(lambda arr: torch.from_numpy(arr.transpose(2, 0, 1)), (img_lq, img_hq))

        return {"gt": img_hq, "lq": img_lq}
