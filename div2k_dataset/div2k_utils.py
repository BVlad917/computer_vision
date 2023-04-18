import os
import cv2
import albumentations as A

# transforms to apply to images BEFORE applying BSRGAN+ degradation model
transforms = A.OneOf([
    A.Rotate(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
], p=0.2)


def download_div2k_data(is_train, is_hr=True, down_sample_mode=None, down_sample_factor=None, data_dir="./div2k_data/"):
    """
    Download DIV2K data subsets.
    :param is_train: if the train subset should be downloaded or the valid subset; boolean
    :param is_hr: if the high-resolution subset should be downloaded or the low-resolution counterpart; boolean
    :param down_sample_mode: what down-sampling strategy to have used; string (e.g., "bicubic")
    :param down_sample_factor: what down-sampling factor to have used; int (e.g., 2/4/8)
    :param data_dir: where to save the data; string
    :return: the path to where the data was just saved
    """
    if is_hr:
        assert down_sample_mode is None, "Cannot have down-sampling mode when working with the HQ images of DIV2K"
        assert down_sample_factor is None, "Cannot have down-sampling factor when working with the HQ images of DIV2K"
    else:
        assert down_sample_mode is not None, "Must have down-sampling mode if working with LQ images"
        assert down_sample_factor is not None, "Must have down-sampling factor if working with LQ images"

    train_or_valid = "train" if is_train else "valid"
    hr_or_lr = "HR" if is_hr else "LR"

    data_source = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"  # link to DIV2K data

    # construct the exact data file we want, based on the given function arguments
    file_name = "DIV2K_"
    file_name += train_or_valid + "_"  # train/valid
    file_name += hr_or_lr  # HR/LR
    if not is_hr:
        if down_sample_factor != 8:
            file_name += f"_{down_sample_mode}"  # bicubic
        file_name += f"_X{down_sample_factor}"  # 2/4/8
    file_name += ".zip"

    # construct the 3 commands we need: (1) to download the data, (2) to extract it, and (3) to remove the zip file
    get_cmd = f"wget {data_source}{file_name}"
    unzip_cmd = f"unzip -q {file_name} -d {data_dir}"
    rm_cmd = f"rm {file_name}"

    # run the 3 commands
    download_return_code = os.system(get_cmd)
    if download_return_code == 0:
        # if the download was successful => unzip the file and remove the archive
        os.system(unzip_cmd)
        os.system(rm_cmd)

    # return the path to the recently downloaded data
    return os.path.join(data_dir, file_name.split('.')[0])
