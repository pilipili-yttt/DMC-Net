import torch
from torch import nn as nn
from torch.nn import functional as F
from Loss.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(EdgeLoss, self).__init__()

        k = torch.tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        kernel_2d = torch.matmul(k.t(), k)

        self.register_buffer(
            'kernel_base',
            kernel_2d.view(1, 1, 5, 5)
        )

        self.weight = loss_weight

    def conv_gauss(self, img):
        B, C, H, W = img.shape

        kernel = self.kernel_base.repeat(C, 1, 1, 1)
        kernel = kernel.to(device=img.device, dtype=img.dtype)  # ✅ 关键修复

        img = F.pad(img, (2, 2, 2, 2), mode='replicate')
        return F.conv2d(img, kernel, groups=C)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)

        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4

        filtered = self.conv_gauss(new_filter)
        return current - filtered

    def forward(self, x, y, **kwargs):
        loss = F.l1_loss(
            self.laplacian_kernel(x),
            self.laplacian_kernel(y)
        )
        return loss * self.weight


class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x-y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class MIX1Loss(nn.Module):
    def __init__(self, loss_weight=0.1):
        super(MIX1Loss, self).__init__()
        self.l1_loss = L1Loss()
        self.fft_loss = FFTLoss(loss_weight = loss_weight)
    def forward(self,pred, target):
        loss = self.l1_loss(pred,target) + self.fft_loss(pred,target)
        return loss

class MIX2Loss(nn.Module):
    def __init__(self, loss_weight=[0.01,0.05]):
        super(MIX2Loss, self).__init__()
        self.CharbonnierLoss = CharbonnierLoss()
        self.fft_loss = FFTLoss(loss_weight=loss_weight[0])
        self.edge_loss = EdgeLoss(loss_weight=loss_weight[1])
    def forward(self,pred, target):
        loss = self.CharbonnierLoss(pred,target) + self.fft_loss(pred,target) + self.edge_loss(pred,target)
        return loss

class MIX3Loss(nn.Module):
    def __init__(self, loss_weight=0.01):
        super(MIX3Loss, self).__init__()
        self.CharbonnierLoss = CharbonnierLoss()
        self.MSELoss = MSELoss(loss_weight=loss_weight)
        self.fft_loss = FFTLoss(loss_weight=loss_weight)
    def forward(self,pred, target):
        loss = self.CharbonnierLoss(pred,target) + self.fft_loss(pred,target)
        return loss
