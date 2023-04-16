import cv2
import albumentations as A

# transforms to apply to images BEFORE applying BSRGAN+ degradation model
transforms = A.Compose([
    A.Rotate(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])
