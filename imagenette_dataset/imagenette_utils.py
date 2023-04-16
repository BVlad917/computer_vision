import os
import shutil
import albumentations as A
from fastai.vision.all import URLs, untar_data
from albumentations.pytorch.transforms import ToTensorV2


# the ImageNette classes are by default encoded with codes such as "n02102040" instead of simply "English springer"
LBL_2_STRING = dict(
    n01440764='tench',
    n02102040='English springer',
    n02979186='cassette player',
    n03000684='chain saw',
    n03028079='church',
    n03394916='French horn',
    n03417042='garbage truck',
    n03425413='gas pump',
    n03445777='golf ball',
    n03888257='parachute',
)

# label to int for ImageNette classes (e.g., 0: "tench")
LBL_2_INT = {k: idx for idx, (k, _) in enumerate(LBL_2_STRING.items())}

# int to label for ImageNette classes (e.g., 0: "n01440764")
INT_2_LBL = {idx: k for idx, (k, _) in enumerate(LBL_2_STRING.items())}

# dir with link of the dataset types to their directory name
# (e.g, 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz': 'imagenette2-160')
DATA_LINK_2_NAME = {
    p: os.path.basename(p).split('.')[0] for p in [URLs.IMAGENETTE_160, URLs.IMAGENETTE_320, URLs.IMAGENETTE]
}


def get_imagenette_data(to_download=URLs.IMAGENETTE):
    if DATA_LINK_2_NAME[to_download] in os.listdir("."):
        return
    path = untar_data(URLs.IMAGENETTE_160)  # extract the data
    path = str(path)  # get the path as a string
    new_path = shutil.move(path, ".")  # move from default fastai location to working dir
    return new_path


train_transforms = A.Compose([
    A.LongestMaxSize(max_size=320, interpolation=1),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0, 0, 0)),

    A.OneOf([
        A.ColorJitter(),
        A.Blur(blur_limit=3),
    ]),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.HorizontalFlip(),

    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
    A.RandomBrightnessContrast(p=0.5),

    A.RandomCrop(height=224, width=224),
    A.Normalize(),  # divides by 255 and then subtracts the mean and divides by the std of ImageNet
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.LongestMaxSize(max_size=320, interpolation=1),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0, 0, 0)),

    A.CenterCrop(height=224, width=224),
    A.Normalize(),  # divides by 255 and then subtracts the mean and divides by the std of ImageNet
    ToTensorV2(),
])
