import torch
from typing import Optional,List,Dict

def per_label_dice_coefficient(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    labels: Optional[List[int]] = None
) -> Dict[int, float]:
    """
    Calculate per-label Dice coefficient for multi-class segmentation.
    Works with 2D (H, W), 3D (D, H, W), and batched inputs.
    
    Args:
        y_true: Ground truth labels 
                - 2D: (H, W)
                - 3D: (D, H, W) 
                - Batched 2D: (N, H, W)
                - Batched 3D: (N, D, H, W)
        y_pred: Predicted labels (same shape as y_true)
        labels: List of label values to compute Dice for. If None, uses all unique labels.
        eps: Smoothing factor to avoid division by zero
        
    Returns:
        Dictionary with label as key and Dice coefficient as value
    """
    # Ensure tensors are on same device
    if y_true.device != y_pred.device:
        y_pred = y_pred.to(y_true.device)
    
    # Check shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Get unique labels if not provided
    if labels is None:
        labels: List = [0,1,2,3]
    
    dice_scores = {}
    
    for label in range(4):
        if label in labels:
            # Create binary masks for current label
            true_mask = (y_true == label).float()
            pred_mask = (y_pred == label).float()
            
            # Calculate intersection and union
            intersection = torch.sum(true_mask * pred_mask)
            total = torch.sum(true_mask) + torch.sum(pred_mask)
            
            if total != 0:
                dice = (2.0 * intersection) / (total)
            else:
                dice = torch.tensor(-1,device=y_true.device)
            dice_scores[int(label)] = float(dice.item())
        else:
            dice = torch.tensor(-1,device=y_true.device)
            dice_scores[int(label)] = float(dice.item())
    
    return dice_scores

'''
def stupid_dice_coefficient(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    labels: Optional[List[int]] = None
) -> Dict[int, float]:
    """
    Calculate per-label Dice coefficient for multi-class segmentation.
    Works with 2D (H, W), 3D (D, H, W), and batched inputs.
    
    Args:
        y_true: Ground truth labels 
                - 2D: (H, W)
                - 3D: (D, H, W) 
                - Batched 2D: (N, H, W)
                - Batched 3D: (N, D, H, W)
        y_pred: Predicted labels (same shape as y_true)
        labels: List of label values to compute Dice for. If None, uses all unique labels.
        eps: Smoothing factor to avoid division by zero
        
    Returns:
        Dictionary with label as key and Dice coefficient as value
    """
    # Ensure tensors are on same device
    if y_true.device != y_pred.device:
        y_pred = y_pred.to(y_true.device)
    
    # Check shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Get unique labels if not provided
    if labels is None:
        labels: List = [0,1,2,3]
    
    dice_scores = {}
    
    for label in range(4):
        if label in labels:
            # Create binary masks for current label
            true_mask = (y_true == label).float()
            pred_mask = (y_pred == label).float()
            
            # Calculate intersection and union
            intersection = torch.sum(true_mask * pred_mask)
            total = torch.sum(true_mask) + torch.sum(pred_mask)
            
            if total != 0:
                dice = (2.0 * intersection) / (total)
            else:
                dice = torch.tensor(-1,device=y_true.device)
            dice_scores[int(label)] = float(dice.item())
        else:
            dice = torch.tensor(-1,device=y_true.device)
            dice_scores[int(label)] = float(dice.item())
    
    # Stupid part
    for labels in range(1,4):
        
    return dice_scores
'''