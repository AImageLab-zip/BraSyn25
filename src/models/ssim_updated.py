import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    """Create 1D Gaussian kernel"""
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel, device=None, dtype=torch.float32):
    """Create 2D Gaussian window for SSIM computation"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    
    # Expand to match number of channels
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    
    # Move to specified device and dtype
    if device is not None:
        window = window.to(device=device, dtype=dtype)
    
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Core SSIM computation"""
    # Ensure window is on correct device and has correct dtype
    window = window.to(device=img1.device, dtype=img1.dtype)
    
    padding = window_size // 2
    
    # Compute local means
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    """SSIM Loss Module with improved memory management"""
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        # Don't pre-create window - create it dynamically to avoid device issues
        self.register_buffer('_window_cache', torch.empty(0))
        self._cached_channel = 0
        
    def _get_window(self, channel, device, dtype):
        """Get or create window with proper device/dtype"""
        if (self._cached_channel != channel or 
            self._window_cache.device != device or 
            self._window_cache.dtype != dtype or
            self._window_cache.numel() == 0):
            
            # Create new window
            window = create_window(self.window_size, channel, device, dtype)
            # Cache it
            self._window_cache = window
            self._cached_channel = channel
        
        return self._window_cache

    def forward(self, img1, img2):
        """Forward pass"""
        if img1.shape != img2.shape:
            raise ValueError(f"Input images must have the same shape. Got {img1.shape} and {img2.shape}")
            
        channel = img1.size(1)
        window = self._get_window(channel, img1.device, img1.dtype)
        
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIMLoss(nn.Module):
    """SSIM Loss (1 - SSIM for minimization)"""
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(window_size, size_average)
    
    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)


def ssim(img1, img2, window_size=11, size_average=True):
    """Functional interface for SSIM"""
    if img1.shape != img2.shape:
        raise ValueError(f"Input images must have the same shape. Got {img1.shape} and {img2.shape}")
        
    channel = img1.size(1)
    window = create_window(window_size, channel, img1.device, img1.dtype)
    return _ssim(img1, img2, window, window_size, channel, size_average)


# Example usage and testing
if __name__ == "__main__":
    # Test the implementation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test images
    img1 = torch.randn(1, 3, 256, 256).to(device)
    img2 = torch.randn(1, 3, 256, 256).to(device)
    
    # Test functional interface
    ssim_value = ssim(img1, img2)
    print(f"SSIM value: {ssim_value.item():.4f}")
    
    # Test module interface
    ssim_module = SSIM().to(device)
    ssim_value_module = ssim_module(img1, img2)
    print(f"SSIM module value: {ssim_value_module.item():.4f}")
    
    # Test loss
    ssim_loss = SSIMLoss().to(device)
    loss_value = ssim_loss(img1, img2)
    print(f"SSIM loss value: {loss_value.item():.4f}")