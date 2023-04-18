import torch
from torch import nn


class PSNR(nn.Module):
    def __init__(self, crop_border):
        super().__init__()
        self.crop_border = crop_border

    def forward(self, lr, gt):
        psnr = self.psnr_torch(lr=lr, gt=gt, crop_border=self.crop_border)
        return psnr

    @staticmethod
    def psnr_torch(lr, gt, crop_border):
        """Implementation of PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function.
        Supports PSNR across a batch.
        Args:
            lr (torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
            gt (torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]
            crop_border (int): crop border a few pixels
        Returns:
            psnr_metrics (torch.Tensor): PSNR metrics
        """
        # Check if two tensor scales are similar
        assert lr.shape == gt.shape, f"Supplied images have different sizes {str(lr.shape)} and {str(gt.shape)}"

        # crop border pixels
        if crop_border > 0:
            lr = lr[:, :, crop_border:-crop_border, crop_border:-crop_border]
            gt = gt[:, :, crop_border:-crop_border, crop_border:-crop_border]

        # Convert data type to torch.float64 bit
        lr = lr.to(torch.float64)
        gt = gt.to(torch.float64)

        # Calculate PSNR
        mse_value = torch.mean((lr * 255.0 - gt * 255.0) ** 2 + 1e-8, dim=[-1, -2, -3])  # replace with -1, -2, -3
        psnr_metrics = 10 * torch.log10_(255.0 ** 2 / mse_value)

        return psnr_metrics


all_psnr = torch.empty(0)
psnr_model = PSNR(crop_border=0)

a = torch.rand(4, 3, 24, 24)
b = torch.rand(4, 3, 24, 24)
psnr_batch1 = psnr_model(lr=b, gt=a)
all_psnr = torch.cat((all_psnr, psnr_batch1))

a = torch.rand(4, 3, 24, 24)
b = torch.rand(4, 3, 24, 24)
psnr_batch2 = psnr_model(lr=b, gt=a)
all_psnr = torch.cat((all_psnr, psnr_batch2))

print(all_psnr.mean().item())
