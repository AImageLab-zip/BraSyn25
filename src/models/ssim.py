import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

# Function that returns a 2d gaussian kernel
def create_window(window_size, channel): 
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).clone().contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel):
    window = window.to(img1.device).type_as(img1) 
    padding = (window_size -1)//2
    mu1 = F.conv2d(img1, window, padding = padding, groups = channel)
    mu2 = F.conv2d(img2, window, padding = padding, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = padding, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = padding, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = padding, groups = channel) - mu1_mu2

    if img1.numel() > 0: # Check if tensor is not empty
        min_val = img1.min()
        max_val = img1.max()
        dynamic_L = max_val - min_val
        if dynamic_L == 0: # Handle flat images that shouldn't be present in the dataset
            dynamic_L = 5.0 # Or a small epsilon like 1e-6 if images are truly flat but different
    else:
        dynamic_L = 5.0 # Good approximation that will never be used

    # Then use this dynamic_L to calculate C1 and C2 for the current call
    C1 = (0.01 * dynamic_L)**2
    C2 = (0.03 * dynamic_L)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map
    

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11,channels = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channels
        self.window = create_window(window_size, self.channel) # 2d gaussian kernel creation

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            raise RuntimeError('Wrong windows size')
        return _ssim(img1, img2, window, self.window_size, channel)

def ssim(img1, img2, window_size = 11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    return _ssim(img1, img2, window, window_size, channel)
