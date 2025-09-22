import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def show_image_mask(image: torch.Tensor, segmentation: torch.Tensor,gt_segmentation:torch.Tensor):
    if image.shape != segmentation.shape:
        raise RuntimeError(f'The two tensors must have the same shape, but found: {image.shape} and {segmentation.shape}')

    # Define colormap for segmentation
    cmap = ListedColormap([
    (0, 0, 0, 0),    # 0 → completamente trasparente
    (1, 0, 0, 1),    # 1 → red
    (0, 1, 0, 1),    # 2 → lime
    (0, 0, 1, 1)     # 3 → blue
    ])


    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    # Original image
    axes[0].imshow(image.cpu().numpy(), cmap='gray')
    axes[0].set_title("Original Brain")
    axes[0].axis('off')

    # Image with segmentation overlay
    axes[1].imshow(image.cpu().numpy(), cmap='gray')
    axes[1].imshow(segmentation.cpu().numpy(), cmap=cmap, alpha=0.2)
    axes[1].set_title("Brain with segmentation")
    axes[1].axis('off')

    # Image with gt_segmentation overlay
    axes[2].imshow(image.cpu().numpy(), cmap='gray')
    axes[2].imshow(gt_segmentation.cpu().numpy(), cmap=cmap, alpha=0.2)
    axes[2].set_title("Brain with GT segmentation")
    axes[2].axis('off')

    plt.tight_layout()
    return fig

def show_image_mask_four(image: torch.Tensor, segmentation_1: torch.Tensor,segmentation_2: torch.Tensor,gt_segmentation:torch.Tensor,
                         label_1:str = 'segmentation_1',label_2:str = 'segmentation_2'):
    if image.shape != segmentation_1.shape !=segmentation_2.shape:
        raise RuntimeError(f'The two tensors must have the same shape, but found: {image.shape} and {segmentation_1.shape} and {segmentation_2.shape}')

    # Define colormap for segmentation
    cmap = ListedColormap([
    (0, 0, 0, 0),    # 0 → completamente trasparente
    (1, 0, 0, 1),    # 1 → red
    (0, 1, 0, 1),    # 2 → lime
    (0, 0, 1, 1)     # 3 → blue
    ])


    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(18, 10))

    # Original image
    axes[0].imshow(image.cpu().numpy(), cmap='gray')
    axes[0].set_title("Original Brain")
    axes[0].axis('off')

    # Image with segmentation_1 overlay
    axes[1].imshow(image.cpu().numpy(), cmap='gray')
    axes[1].imshow(segmentation_1.cpu().numpy(), cmap=cmap, alpha=0.2)
    axes[1].set_title(label_1)
    axes[1].axis('off')
    
    # Image with segmentation_1 overlay
    axes[2].imshow(image.cpu().numpy(), cmap='gray')
    axes[2].imshow(segmentation_2.cpu().numpy(), cmap=cmap, alpha=0.2)
    axes[2].set_title(label_2)
    axes[2].axis('off')

    # Image with gt_segmentation overlay
    axes[3].imshow(image.cpu().numpy(), cmap='gray')
    axes[3].imshow(gt_segmentation.cpu().numpy(), cmap=cmap, alpha=0.2)
    axes[3].set_title("Brain with GT segmentation")
    axes[3].axis('off')

    plt.tight_layout()
    return fig

