import torch
from torch import nn


class PSNR(nn.Module):
    def __init__(self, crop_border):
        super().__init__()
        self.crop_border = crop_border

    def forward(self, sr, gt):
        psnr = self.psnr_torch(sr=sr, gt=gt, crop_border=self.crop_border)
        return psnr

    @staticmethod
    def psnr_torch(sr, gt, crop_border):
        """Implementation of PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function.
        Supports PSNR across a batch.
        Args:
            sr (torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
            gt (torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]
            crop_border (int): crop border a few pixels
        Returns:
            psnr_metrics (torch.Tensor): PSNR metrics
        """
        # Check if two tensor scales are similar
        assert sr.shape == gt.shape, f"Supplied images have different sizes {str(sr.shape)} and {str(gt.shape)}"

        # crop border pixels
        if crop_border > 0:
            sr = sr[:, :, crop_border:-crop_border, crop_border:-crop_border]
            gt = gt[:, :, crop_border:-crop_border, crop_border:-crop_border]

        # Convert data type to torch.float64 bit
        sr = sr.to(torch.float64)
        gt = gt.to(torch.float64)

        # Calculate PSNR
        mse_value = torch.mean((sr * 255.0 - gt * 255.0) ** 2 + 1e-8, dim=[-1, -2, -3])  # replace with -1, -2, -3
        psnr_metrics = 10 * torch.log10_(255.0 ** 2 / mse_value)

        return psnr_metrics
