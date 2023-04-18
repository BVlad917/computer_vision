import os
import glob
from torch.utils.data import Dataset
from torchvision.io import read_image


class Div2KDatasetPreLoaded(Dataset):
    def __init__(self, path, img_ext="png"):
        self.gt_imgs = glob.glob(os.path.join(path, "gt", f"*.{img_ext}"))

    def __getitem__(self, index):
        gt_path = self.gt_imgs[index]
        lq_path = gt_path.replace("gt", "lq")
        gt, lq = map(lambda path: read_image(path), (gt_path, lq_path))
        return {"gt": gt, "lq": lq}

    def __len__(self):
        return len(self.gt_imgs)
