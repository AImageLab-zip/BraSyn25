import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainTumorLoss(nn.Module):
    """Combined CE + Dice Loss for brain tumor segmentation"""
    
    def __init__(self, ce_weight=0.5, dice_weight=0.5, smooth=1e-6, class_weights=None):
        super(BrainTumorLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
        # Set class weights for CE (handle class imbalance)
        if class_weights is not None:
            self.class_weights = class_weights
        else:
            raise RuntimeError('please provide class weights for BrainTumorLoss init')
    
    def dice_loss(self, pred_logits, target):
        """Multi-class Dice Loss"""
        pred_softmax = F.softmax(pred_logits, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=pred_logits.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate dice for each class
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice = dice * self.class_weights

        return 1 - dice.mean()
    
    def forward(self, pred_logits, target):
        # Cross entropy loss (expects logits, not softmax)
        ce_loss = F.cross_entropy(pred_logits, target.long().squeeze(1), weight=self.class_weights)
        
        # Dice loss
        dice_loss = self.dice_loss(pred_logits, target.squeeze(1))
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return total_loss, ce_loss, dice_loss
    


class FocalWeightedDiceCELoss2D(nn.Module):
    def __init__(self, gamma=2.0, smooth=1e-3, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] - raw logits from model
        targets: [B, 1, H, W] - class indices (long tensor)
        """
        # Input validation
        assert logits.dim() == 4, f"Expected 4D logits, got {logits.dim()}D, shape --> {logits.shape}"
        assert targets.dim() == 4, f"Expected 4D targets, got {targets.dim()}D, shape --> {targets.shape}"
        assert logits.shape[0] == targets.shape[0], "Batch size mismatch"
        assert logits.shape[2:] == targets.shape[2:], "Spatial dimensions mismatch"
        
        num_classes = logits.shape[1]
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        
        # Create one-hot encoded targets
        targets_one_hot = F.one_hot(targets.long().squeeze(1), num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Extract predicted probabilities for the true classes (p_t)
        p_t = (probs * targets_one_hot).sum(dim=1, keepdim=True)  # [B, 1, H, W] 
        
        # Compute focal weights per pixel
        focal_weights = (1 - p_t) ** self.gamma  # [B, 1, H, W]
        
        # ==================== DICE LOSS ====================
        # Compute Dice coefficient for each class
        dims = (0, 2, 3)  # Sum over batch, height, width
        intersection = (probs * targets_one_hot).sum(dims)  # [C]
        union = (probs + targets_one_hot).sum(dims)  # [C]
        
        # Compute Dice score with smoothing
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [C]
        dice_score = ((1 - dice_score) ** self.gamma) * dice_score # [c]
        dice_loss = 1.0 - dice_score.mean()
        
        # ==================== FOCAL CE LOSS ====================
        # Compute cross entropy for each pixel (multi-class)
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, H, W]
        ce_loss = -(targets_one_hot * log_probs).sum(dim=1)  # [B, H, W]
        
        # Apply focal weighting to CE loss
        focal_weights_squeezed = focal_weights.squeeze(1)  # [B, H, W]
        focal_ce_loss = focal_weights_squeezed * ce_loss  # [B, H, W]
        
        # Average over spatial dimensions and batch
        focal_ce_loss = focal_ce_loss.mean()
        
        # ==================== COMBINE LOSSES ====================
        total_loss = self.dice_weight * dice_loss + self.bce_weight * focal_ce_loss
        
        return total_loss, dice_loss, focal_ce_loss



import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        """
        Focal Tversky Loss for extreme class imbalance
        
        Args:
            alpha: controls false negatives (lower = more focus on recall)
            beta: controls false positives (higher = more focus on precision) 
            gamma: focal parameter (higher = more focus on hard examples)
            smooth: smoothing factor
            
        For 98% imbalance, typically:
        - alpha=0.3, beta=0.7 (emphasize recall over precision)
        - gamma=1.33 or higher
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

        
    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] - raw logits 
        targets: [B, 1, H, W] - class indices
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets.long().squeeze(1), num_classes).permute(0, 3, 1, 2).float()
        
        # Compute per-class Tversky index
        dims = (0, 2, 3)  # batch, height, width
        
        # True positives, false positives, false negatives
        tp = (probs * targets_one_hot).sum(dims)  # [C]
        fp = (probs * (1 - targets_one_hot)).sum(dims)  # [C]
        fn = ((1 - probs) * targets_one_hot).sum(dims)  # [C]
        
        # Tversky index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Focal Tversky loss
        focal_tversky = (1 - tversky_index) ** self.gamma
        return focal_tversky.mean(), focal_tversky, None

class CombinedFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, focal_gamma=2.0, 
                 tversky_weight=0.7, focal_weight=0.3):
        super().__init__()
        self.focal_tversky = FocalTverskyLoss(alpha, beta, gamma)
        self.focal_gamma = focal_gamma
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        
    def forward(self, logits, targets):
        # Focal Tversky loss
        ft_loss, _, _ = self.focal_tversky(logits, targets)
        
        # Standard focal CE loss
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets.long().squeeze(1), num_classes).permute(0, 3, 1, 2).float()
        
        # CE loss with focal weighting
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Get probability of true class for focal weighting
        p_t = (probs * targets_one_hot).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        focal_weights = (1 - p_t) ** self.focal_gamma
        
        ce_loss = -(targets_one_hot * log_probs).sum(dim=1)  # [B, H, W]
        focal_ce_loss = (focal_weights.squeeze(1) * ce_loss).mean()
        
        return self.tversky_weight * ft_loss + self.focal_weight * focal_ce_loss, ft_loss, focal_ce_loss