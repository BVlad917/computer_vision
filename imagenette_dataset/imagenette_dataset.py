import os
import cv2
import glob
from torch.utils.data import Dataset

from imagenette_dataset.imagenette_utils import LBL_2_INT


class ImageNetteDataset(Dataset):
    def __init__(self, path, transform=None, img_ext="png"):
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(path, f"**/*.{img_ext}"), recursive=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # get the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply transform on the image
        if self.transform:
            image = self.transform(image=image)["image"]

        # get the image's label
        label_encoding = os.path.basename(os.path.dirname(image_path))
        label = LBL_2_INT[label_encoding]

        return {"image": image, "label": label}
