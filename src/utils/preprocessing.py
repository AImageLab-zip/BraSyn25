import nibabel as nib
import gzip
import torch
from pathlib import Path
import io
import numpy as np
def path_to_nii_to_tensor(path:Path):
    
    nii_img = nib.load(path)
    nii_data = np.ascontiguousarray(nii_img.get_fdata(dtype=np.float32))
    nii_tensor = torch.from_numpy(nii_data)
    
    return nii_tensor

def get_tensor_header_affine_from_nii(path:Path):
    
    nii_img = nib.load(path)
    nii_data = np.ascontiguousarray(nii_img.get_fdata(dtype=np.float32))
    nii_tensor = torch.from_numpy(nii_data).to(torch.float32)
    nii_header = nii_img.header
    nii_affine = nii_img.affine
    return nii_tensor, nii_header, nii_affine

def hard_coded_clamp(brains:torch.Tensor):
    upper_bounds = [4623, 3503, 1797, 2532]
    for i in range(1,5):
        brains[i] = brains[i].clamp(min = 0, max=upper_bounds[i - 1])
    return brains

def swap_segmentations(segmentation:torch.Tensor):
    two_mask = segmentation == 2
    one_mask = segmentation == 1
    segmentation[two_mask] = 1
    segmentation[one_mask] = 2