import torch
# pyright: reportPrivateImportUsage=false
import nibabel as nib
import numpy as np

def save_tensor_to_nii(tensor, filename, affine=None, header = None):
    # Check shape
    if tensor.shape != (1, 240, 240, 155):
        raise ValueError(f"Expected shape (1,240,240,155), got {tensor.shape}")

    # Remove batch dimension and convert to numpy
    np_array = tensor.squeeze(0).cpu().numpy()
    if affine is None:
        affine = np.eye(4)

    if header is None:
        nii_img = nib.Nifti1Image(np_array, affine)
    else:
        nii_img = nib.Nifti1Image(np_array, affine, header)


    # Save to file
    nib.save(nii_img, filename)
    #print(f"Saved to {filename}")