import os 
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from torch.amp.grad_scaler import GradScaler  # Added for AMP
from torch.amp.autocast_mode import autocast
import numpy as np
from tqdm import tqdm
from src.models.seg_unet import BrainTumorLoss, FocalWeightedDiceCELoss2D,CombinedFocalTverskyLoss,FocalTverskyLoss
from pathlib import Path
from src.data.dataset import brain_coll_seg_total_fusion, BrainDataset2D
import wandb 
from typing import List, Dict, Any, Tuple
from dynamic_network_architectures.architectures.unet import PlainConvUNet
import random
from src.utils.metrics import per_label_dice_coefficient

from src.utils.showing import show_image_mask
import torch
from src.utils.preprocessing import path_to_nii_to_tensor
from matplotlib import pyplot as plt
from src.utils.preprocessing import swap_segmentations
from argparse import ArgumentParser


class BrainTumorTrainer:
    def __init__(self, model, train_loaders:List[DataLoader], val_loader, device, 
                 name:str, pretrain_fraction:float,
                 lr=2e-3, ce_weight=0.5, dice_weight=0.5, class_weights=None, use_amp=True,num_epochs=100
                 ):
        self.model = model.to(device,memory_format=torch.channels_last)
        self.train_loader_pretrain = train_loaders[0]
        self.train_loader_finetune = train_loaders[1]
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'  # Only use AMP on CUDA
        self.start_epoch = 0
        self.num_epochs = num_epochs
        self.set_seed()
        self.epoch = 0
        self.name = name
        self.pretrain_fraction = pretrain_fraction

        # Initialize GradScaler for AMPmemory_format = torch.channels_first
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP)")
        else:
            self.scaler = None
            print("Training without AMP")
        
        # Loss function
        '''self.criterion = BrainTumorLoss(ce_weight=ce_weight, 
                                     dice_weight=dice_weight, 
                                     class_weights=class_weights).to(device)'''
        #self.criterion = CombinedFocalTverskyLoss(class_weights=False).to(device)
        self.criterion = FocalTverskyLoss().to(device)
        # Optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = PolynomialLR(self.optimizer, total_iters=num_epochs)
        
        
        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.train_dice = []
        self.val_dice = []
        self.learning_rates = []
        
        # Initialize wandb with proper error handling
        try:
            wandb.init(
                dir=os.path.join(os.path.expanduser("~"), ".wandb_logs"),
                project="HF_STUFF",
                name=self.name,
                reinit=True
            )
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            self.use_wandb = False
        else:
            self.use_wandb = True

    def set_seed(self, seed:int = 0):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


    def train_epoch(self):
        """Train for one epoch with AMP support"""
        self.model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_dice_loss = 0.0
        
        pbar = tqdm(self.train_loader_pretrain, desc="Pretraining") if self.epoch < self.num_epochs * self.pretrain_fraction else tqdm(self.train_loader_finetune, desc="Finetuning")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                data = batch['image'].to(self.device, memory_format = torch.channels_last ,non_blocking=True)
                target = batch['segmentations'].to(self.device, memory_format = torch.channels_last , non_blocking=True)
                
                self.optimizer.zero_grad()
                
                # Forward pass with autocast
                
                if self.use_amp:
                    with autocast(device_type="cuda"):
                        output = self.model(data)
                        total_loss, per_label_loss, _ = self.criterion(output, target)
                else:
                    output = self.model(data)
                    total_loss, per_label_loss, _ = self.criterion(output, target)
                
                # Check for invalid loss values
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: Invalid loss detected at batch {batch_idx}")
                    continue
                wandb.log({
                    'train/total_loss':total_loss
                })
                # Backward pass with gradient scaling
                if self.use_amp and self.scaler is not None:
                    # Scale loss and backward pass
                    self.scaler.scale(total_loss).backward()
                    
                    # Step optimizer with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard backward pass
                    total_loss.backward()
                    self.optimizer.step()
                
                for i in range(4):
                    wandb.log({f'train_tversky_{i}':per_label_loss[i]})
                
                '''
                # Calculate Dice coefficient using argmax of logits
                with torch.no_grad():  # No gradients needed for metrics
                    pred_classes = torch.argmax(output.float(), dim=1) # Convert to float32
                    batch_dice = per_label_dice_coefficient(target.squeeze(1),pred_classes)'''
                
                        
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        
        return 0.0, 0.0, 0.0, 0.0
    
    def validate_epoch(self)-> Tuple[Any,Any,Any,Dict]:
        """Validate for one epoch with AMP support"""
        self.model.eval()
        running_loss = 0.0
        running_bce = 0.0
        running_dice_loss = 0.0
        total_dice = {
            key: [] for key in range(4)
        }
        total_iou = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(pbar):
                data = batch['image'].to(self.device, non_blocking=True,memory_format = torch.channels_last)
                target = batch['segmentations'].to(self.device, memory_format = torch.channels_last ,non_blocking=True)
                #swap_segmentations(target)
                # Forward pass with autocast (even in validation for consistency)
                
                if self.use_amp:
                    with autocast(device_type="cuda"):
                        output = self.model(data)
                        total_loss, _, _ = self.criterion(output, target)
                else:
                    output = self.model(data)
                    total_loss, _, _ = self.criterion(output, target)
                
                # Check for invalid loss values
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: Invalid validation loss at batch {batch_idx}")
                    continue
                
                # Statistics (convert to float32)
                running_loss += total_loss.float().item()
                running_bce += 0
                running_dice_loss += 0
                
                # Calculate metrics using argmax of logits
                pred_classes = torch.argmax(output.float(), dim=1).squeeze(1)
                batch_dice = per_label_dice_coefficient(target.squeeze(1),pred_classes,labels=[0,1,2,3])
                try:
                    for i in range(4):
                        if batch_dice[i] >= 0.0:
                            total_dice[i].append(batch_dice[i])
                except:
                    print(batch_dice)
                
                pbar.set_postfix({
                    'Val_Loss': f'{total_loss.float().item():.4f}'
                })
                    
        
        # Calculate epoch statistics
        if len(self.val_loader) > 0:
            epoch_loss = running_loss / len(self.val_loader)
            epoch_bce = running_bce / len(self.val_loader)
            epoch_dice_loss = running_dice_loss / len(self.val_loader)
            epoch_dice ={
                key:np.mean(value) for key,value in total_dice.items()
            }
            epoch_iou = np.mean(total_iou) if total_iou else 0.0
        else:
            epoch_loss = epoch_bce = epoch_dice_loss =  epoch_iou = 0.0
            epoch_dice = {
                key: 0.0 for key in range(4)
            }
        
        return epoch_loss, epoch_bce, epoch_dice_loss, epoch_dice
    
    def train(self):
        """Full training loop with comprehensive error handling"""
        save_dir=f"/work_path/checkpoints/{self.name}"
        os.makedirs(save_dir, exist_ok=True)
        num_epochs = self.num_epochs
        
        best_val_dice = 0.0
        patience_counter = 0
        max_patience = 20
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"Train batches pretrain: {len(self.train_loader_pretrain)}")
        print(f"Train batches finetune: {len(self.train_loader_finetune)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        for epoch in range(self.start_epoch, num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training
            train_loss, train_bce, train_dice_loss, train_dice = self.train_epoch()

            # Validation
            val_loss, val_bce, val_dice_loss, val_dice = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dice.append(train_dice)
            self.val_dice.append(val_dice)
            self.learning_rates.append(current_lr)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss}, Val Loss: {val_loss}")
            print(f"Train Dice: {train_dice}, Val Dice: {val_dice}")
            print(f"Learning Rate: {current_lr:.2e}")
            if self.use_amp and self.scaler is not None:
                print(f"Gradient Scale: {self.scaler.get_scale():.0f}")
            
            # Log epoch metrics to wandb
            if self.use_wandb:
                try:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train/epoch_loss': train_loss,
                        'train/epoch_dice': train_dice,
                        'val/epoch_loss': val_loss,
                        'lr': current_lr
                    }
                    wandb.log({
                        f'val/dice_{key}':value for key,value in val_dice.items()
                    })
                    if self.use_amp and self.scaler is not None:
                        log_dict['grad_scale'] = self.scaler.get_scale()
                    wandb.log(log_dict)
                except Exception as e:
                    print(f"Warning: Failed to log epoch metrics: {e}")
            
            # Create checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_dice': best_val_dice,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_dice': self.train_dice,
                'val_dice': self.val_dice,
                'learning_rates': self.learning_rates,
                'use_amp': self.use_amp
            }
            
            # Add scaler state if using AMP
            if self.use_amp and self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        
        print(f"\nTraining completed! Best validation Dice: {best_val_dice:.4f}")
        
        # Close wandb
        if self.use_wandb:
            try:
                wandb.finish()
            except:
                pass
        
        return self.train_losses, self.val_losses, self.train_dice, self.val_dice

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint with AMP support"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_dice = checkpoint.get('train_dice', [])
        self.val_dice = checkpoint.get('val_dice', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        self.start_epoch = checkpoint['epoch']
        return



if __name__ == "__main__":
    # Parser thing
    parser = ArgumentParser()
    parser.add_argument('--identifier',required=True,help='provide a name for the run', type=str)
    parser.add_argument('--pretrain-datasets',required =True,help='provide a pretrain dataset path/a list of them',nargs='+',type=str)
    parser.add_argument('--finetune-datasets',required =True,help='provide a finetune dataset path/a list of them',nargs='+',type=str)
    parser.add_argument('--val-datasets',required =True,help='provide a validation dataset path/a list of them',nargs='+',type=str)
    parser.add_argument('--train-splits',required =True,help='provide a finetune dataset path/a list of them',nargs='+',type=str)
    parser.add_argument('--val-splits',required =True,help='provide a finetune dataset path/a list of them',nargs='+',type=str)
    parser.add_argument('--pretrain-fraction',required = True,
                        help='provide a decimal in the range [0;1] representing the percentage of epochs for pretraining',type=float)
    parser.add_argument('--num-classes',default=4,help='number of segmentation classes (background + tumor classes)',type=int, metavar='N_CLASSES')
    parser.add_argument('--batch-size',default=20,help='batch size for training', type = int, metavar='BATCH_SIZE')
    parser.add_argument('--start-check-index',default = None, type=int)
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        use_amp = capability[0] >= 6
        print(f'cuda capability --> {capability[0]}')
    else:
        use_amp = False
    print(f"AMP supported: {use_amp}")
    
    # Hyperparameters
    batch_size = args.batch_size
    val_batch_size = batch_size*2
    num_workers = 8
    num_epochs = 100
    learning_rate = 1e-3 * (batch_size/48)
    stages = [32, 64, 128, 256]
    # Create model
    model = PlainConvUNet(
        input_channels=9,                   
        n_stages=len(stages),                        
        features_per_stage=stages,
        conv_op=nn.Conv2d,                 # 2D conv layers
        kernel_sizes=3,                    # 3x3 kernels d
        strides=1,                         # stride 1 convs
        n_conv_per_stage=2,                # 2 conv layers per encoder stage
        num_classes=args.num_classes,                
        n_conv_per_stage_decoder=2,        # 2 conv layers per decoder stage
        conv_bias=False,                   # no bias because of batchnorm
        norm_op=nn.BatchNorm2d,            # 2D batch normalization
        norm_op_kwargs={},                 # default norm kwargs
        dropout_op=nn.Dropout2d,           # spatial dropout for 2D
        dropout_op_kwargs={"p": 0.1},      # 10% dropout
        nonlin=nn.SiLU,                    # ReLU activation
        nonlin_kwargs={"inplace": True},   # inplace silu
        deep_supervision=False,            # no deep supervision
        nonlin_first=False                 # nonlinearity after conv+norm
    )
    model.compile(fullgraph = True, mode = 'max-autotune')

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets and data loaders
    train_dataset_pretrain = BrainDataset2D(dataset_dirs = args.pretrain_datasets,
                                   split = args.train_splits,
                                   verbose=True)
    
    train_dataset_finetune = BrainDataset2D(dataset_dirs=args.finetune_datasets,
                                   split = args.train_splits,
                                   verbose=True)
    val_dataset = BrainDataset2D(dataset_dirs=args.val_datasets,split=args.val_splits,
                                 verbose=True)
    
    print(f"Train dataset size: {len(train_dataset_pretrain)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    train_loader_pretrain = DataLoader(
        train_dataset_pretrain, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=brain_coll_seg_total_fusion,
        persistent_workers=True,
        drop_last=True  # Added to avoid issues with batch norm
    )
    
    train_loader_finetune = DataLoader(
        train_dataset_finetune, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=brain_coll_seg_total_fusion,
        persistent_workers=True,
        drop_last=False  # Added to avoid issues with batch norm
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=brain_coll_seg_total_fusion,
        persistent_workers=False
    )
        
    print(f'Pretrain loader len --> {len(train_loader_pretrain)}')
    print(f'Finetune loader len --> {len(train_loader_finetune)}')
    print(f'Validation loader len --> {len(val_loader)}')
    
    # Create trainer with AMP support
    trainer = BrainTumorTrainer(
        model=model,
        train_loaders=[train_loader_pretrain,train_loader_finetune],
        val_loader=val_loader,
        device=device,
        lr=learning_rate,
        ce_weight=0.1,  # Weight for BCE loss
        dice_weight=0.9,  # Weight for Dice loss
        #class_weights=class_weights,
        use_amp=use_amp,  # Enable AMP
        num_epochs=num_epochs,
        name = args.identifier,
        pretrain_fraction=0.3
    )
    if args.start_check_index is not None:
        check_index = args.start_check_index
        while(True):
            try:
                trainer.load_checkpoint(f'/work_path/checkpoints/{args.identifier}/checkpoint_epoch_{check_index}.pth')
                print(f'Loaded checkpoint {check_index}')
                break
            except:
                check_index -=1
                if check_index < 0:
                    print('No checkpoint found')
                    break

    # Train the model
    try:
        print("Starting training...")
        train_losses, val_losses, train_dice, val_dice = trainer.train()
        print("Training completed successfully!")
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"Best validation Dice: {max(val_dice):.4f}")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final val loss: {val_losses[-1]:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
