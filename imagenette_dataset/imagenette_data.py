import os
import cv2
import glob
from torch.utils.data import Dataset, DataLoader

from imagenette_dataset.imagenette_utils import LBL_2_INT


class ImageNetteDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(path, "**/*.JPEG"), recursive=True)

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


def get_dataloader(dset, batch_size=32, shuffle=True, num_workers=0):
    dl = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dl
