# My imports
from src.models.hf_gan import HFGAN, Discriminator
from src.models.ssim import SSIM
from src.utils.scheduling import CosineAnnealingWarmUpRestarts
from src.utils.normalizations import get_real_fake_comparison_graph
from src.data.dataset import BrainDataset2D, brain_coll_dyn_2d_mask_seg
from src.utils.seeding import set_seed 
from src.models.seg_unet import FocalTverskyLoss
from src.utils.metrics import per_label_dice_coefficient
from src.utils.showing import show_image_mask
from src.utils.preprocessing import swap_segmentations
from src.utils.cleaning import clean_accelerate_checkpoint

# Torch imports
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.optim import Adam
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torch.nn.utils import clip_grad_norm_

# Python imports
import random, argparse, itertools, os, traceback, time

# Other stuff
import numpy as np
from accelerate import Accelerator
from pathlib import Path
import wandb
from tqdm import tqdm
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from functools import reduce

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

def main():
    parser = argparse.ArgumentParser(description='BraTS training')
    
    # Important parser arguments
    parser.add_argument('--identifier', type=str, required=True, metavar='N', help='Select the identifier for file name')
    parser.add_argument('--batch-size', type=int,  default=1, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--num-workers', type=int,  default=16, metavar='N', help='num workers for the dataloader')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epoches to train (default: 100)')
    parser.add_argument('--ssim-coefficients', type=int,required=True,default=5,metavar='N',help='coefficient for the ssim in the loss')
    parser.add_argument('--dice-coefficients', type=int,required=True,default=5,metavar='N',help='coefficient for the dice in the loss')
    parser.add_argument('--ssim-mode', type=str,default='whole',metavar='MODE',help="'whole' to use whole volume ssim, 'double' to use healthy-tumor ssim")
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--compile',action='store_true',help='if provided, the models will be compiled')
    parser.add_argument('--skip-shifts',action='store_true',help='if provided, the training will skip unnecessary shifts in the modalities')
    parser.add_argument('--always-three',action='store_true',help='if provided, the network will always get three available modalities')
    parser.add_argument('--mask-background',action='store_true',help='if provided, the generated background will be masked and ignored for gradient calculations')
    parser.add_argument('--grouped_enc',action='store_true',help='if provided, it use a common encoder with grouped convolution for the generator, instead of separate encoders for each modality')
    parser.add_argument('--dataset',type=str, nargs='+', default=['gli'],help='choose one or more of: gli, met, met_add')
    parser.add_argument('--infuse-view',action='store_true',help='if provided, a views tensor will be infused into the latent space of hfgan. The new model uses flash attention!')
    # May be useful 
    parser.add_argument('--resume', action='store_true', help='resume training by loading last snapshot')
    parser.add_argument('--pretrain-weights',type=str,help='if given, the models will be loaded with weights from that directory',default=None)
    parser.add_argument('--skip-cycle',action='store_true',help='if provided, the training will skip the cycle loss part')
    parser.add_argument('--segmenter-weights', type=str, default='/work_path/checkpoints/twersky_finetune_final_perversion/checkpoint_epoch_100.pth',help='path of the weights for the segmenter')
    parser.add_argument('--num-heads',type=int,default=16,help='number of heads for the MHA in the middle layer')
    # May be changed in the future
    parser.add_argument('--checkpoints', type=str, default='/work_path/checkpoints/hf_stuff',help='path of training snapshot')
    parser.add_argument('--view',type=str, nargs='+', default=['axi'],help='choose one or more of: axi, cor, sag')
    # Mostly useless parser arguments
    parser.add_argument('--ch-dim', type=int,  default=64, metavar='N', help='channel dimension for netwrok (default: 64)')
    parser.add_argument('--gradient-accumulation-steps', type=int,  default=1, metavar='N', help='gradient_accumulation_steps for training (default: 1)')
    parser.add_argument('--numlayers', type=int, default=4, metavar='N', help='number of transformer layers(default: 4)')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='number of epoches to log (default: 1)')
    parser.add_argument('--save-freq',type=int,default=1000,metavar='N',help='number of steps before saving')

    args = parser.parse_args()

    set_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    print('start')
    accelerator = Accelerator(gradient_accumulation_steps = args.gradient_accumulation_steps)
    
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    ssim_coeff = args.ssim_coefficients
    tversky_coeff = args.dice_coefficients
    ssim_mode = args.ssim_mode
    skip_shifts = args.skip_shifts
    skip_cycle = args.skip_cycle
    always_three = args.always_three
    seg_weights_path = args.segmenter_weights
    dataset_view = args.view
    dataset = args.dataset
    mask_background = args.mask_background
    grouped_enc = args.grouped_enc
    infuse_view = args.infuse_view
    num_heads = args.num_heads
    if mask_background == True and always_three == False:
        raise RuntimeError('--mask-background must be used only with the --always-three configuration')
    
    train_dataset_list = []
    valid_dataset_list = []

    for d in dataset:
        split_file = f'/work_path/split_files/{d}_train.csv' if d != 'gli' else None
        train_dataset_list.append(BrainDataset2D(dataset_dirs = [Path(f'/work_path/brats2d_{view}/train_{d}') for view in dataset_view], split=split_file))
        split_file = f'/work_path/split_files/{d}_val.csv' if d != 'gli' else None
        s = 'val' if d == 'gli' else 'train'
        valid_dataset_list.append(BrainDataset2D(dataset_dirs = [Path(f'/work_path/brats2d_{view}/{s}_{d}') for view in dataset_view], split=split_file))
    
    train_dataset = reduce(lambda d1, d2: d1 + d2, train_dataset_list)
    valid_dataset = reduce(lambda d1, d2: d1 + d2, valid_dataset_list)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator=train_generator,collate_fn=brain_coll_dyn_2d_mask_seg,persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, generator=val_generator,collate_fn=brain_coll_dyn_2d_mask_seg,persistent_workers=True)

    generator = HFGAN(dim=args.ch_dim, num_inputs=4, num_outputs=1, dim_mults=(1,2,4,8,10), n_layers=args.numlayers, skip=True, blocks=False, grouped_encoder=grouped_enc,infuse_view=infuse_view,num_heads=num_heads)
    discriminator = Discriminator(channels=1, num_filters_last=args.ch_dim)
    
    device = accelerator.device
    
    if tversky_coeff != 0:
        segmenter = PlainConvUNet(
            input_channels=4,                   # 
            n_stages=4,                        # 4 down/up stages
            features_per_stage=[32, 64, 128, 256],
            conv_op=nn.Conv2d,                 # 2D conv layers
            kernel_sizes=3,                    # 3x3 kernels d
            strides=1,                        # stride 1 convs
            n_conv_per_stage=2,                # 2 conv layers per encoder stage
            num_classes=4,                    # binary segmentation
            n_conv_per_stage_decoder=2,       # 2 conv layers per decoder stage
            conv_bias=False,                   # no bias because of batchnorm
            norm_op=nn.BatchNorm2d,            # 2D batch normalization
            norm_op_kwargs={},                 # default norm kwargs
            dropout_op=nn.Dropout2d,           # spatial dropout for 2D
            dropout_op_kwargs={"p": 0.0},     # 0% dropout
            nonlin=nn.SiLU,                   # ReLU activation
            nonlin_kwargs={"inplace": True},  # inplace relu
            deep_supervision=False,            # no deep supervision
            nonlin_first=False                 # nonlinearity after conv+norm
        ).to(device)
        segmenter.load_state_dict(torch.load(seg_weights_path,weights_only=False)['model_state_dict'])
        segmenter.eval()

        tversky_loss = FocalTverskyLoss().to(device)

        if args.compile:
            segmenter = segmenter.compile(mode='max-autotune',fullgraph=True)

    if args.compile:
        #generator.compile(mode='max-autotune',fullgraph=True)
        generator = torch.compile(generator)
        #discriminator.compile(mode='max-autotune',fullgraph=True)

    optimizer = Adam(generator.parameters(), lr=0.0)
    optimizer_D = Adam(discriminator.parameters(), lr=0.0)
    steps_per_epoch = len(train_loader)
    # Shifts
    if skip_shifts:
        shift = [False]
    else:
        shift = [False, True, True, True]
    
    # Division by num_processes is fundamental to make the scheduler "aware" of the presence of multiple gpus
    total_iteration = int((epochs*steps_per_epoch*len(shift))/accelerator.num_processes)
    
    lr = (args.lr * float(accelerator.num_processes) * float(batch_size)) / float(2 * 32)
    
    
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=total_iteration, T_mult=1, eta_max=lr, T_up=100, gamma=0.5)
    scheduler_D = CosineAnnealingWarmUpRestarts(optimizer_D, T_0=total_iteration, T_mult=1, eta_max=lr*0.1, T_up=100, gamma=0.5)
    
    if accelerator.is_main_process:
        print(accelerator.num_processes)
    # Useful for masking the background
    means=torch.tensor([1066.3375,  781.2247,  510.9852,  673.4393]).to(device)
    stds = torch.tensor([1301.7006,  944.3418,  769.4159,  804.3864]).to(device)
    mins = (torch.zeros_like(means) - means)/stds
    
    valid_epochs = args.log_interval

        # Loading pretrain weights, if present
    if args.pretrain_weights is not None:
        try:
            generator.load_state_dict(load_file(Path(args.pretrain_weights)/'model.safetensors'))
            discriminator.load_state_dict(load_file(Path(args.pretrain_weights)/'model_1.safetensors'))
            if accelerator.is_main_process:
                print(f'Loaded pretrain weights',flush=True)
        except Exception as e:
            if accelerator.is_main_process:
                print(f'Tried to load pretrain weights but failed: {e}', flush=True)
    
    # Preparation of the accelerate environment
    generator, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(generator, optimizer, train_loader, valid_loader, scheduler)
    discriminator, optimizer_D, scheduler_D = accelerator.prepare(discriminator, optimizer_D, scheduler_D)
    

            
    if accelerator.is_main_process:
        print(f'Total Iteration: {total_iteration}')
    
    epoch = 0
    iterations = 0
        
    if args.resume:
        if(os.path.exists(os.path.join(args.checkpoints,args.identifier))):
            try:
                if accelerator.is_main_process:
                    clean_accelerate_checkpoint(os.path.join(args.checkpoints,args.identifier))

                time.sleep(10)
                accelerator.wait_for_everyone()
                accelerator.load_state(input_dir=os.path.join(args.checkpoints,args.identifier))

                iterations = scheduler.scheduler.T_cur
                epoch_state_path = os.path.join(args.checkpoints, args.identifier,"current_epoch.pth")
                epoch = torch.load(epoch_state_path,weights_only=False)['epoch']
            except Exception as e:
                iterations = 0
                epoch = 0
                print("Full exception:")
                traceback.print_exc()
    if accelerator.is_main_process:
        print(f'iteration: {iterations} epoch : {epoch}')
    
    loss_adversarial = torch.nn.BCEWithLogitsLoss()
    loss_auxiliary = torch.nn.CrossEntropyLoss()
    metric_psnr = PeakSignalNoiseRatio().to(device)
    metric_siim = StructuralSimilarityIndexMeasure().to(device)
    metric_mssiim = MultiScaleStructuralSimilarityIndexMeasure().to(device)
    loss_ssim = torch.compile(SSIM(window_size=11).to(device)) if args.compile else SSIM(window_size=11).to(device)
    
    cand = [0, 1, 2, 3]
    candidates_all = []
    for L in range(len(cand) + 1):
        if L == 0 or L == 1:
            continue
        for subset in itertools.combinations(cand, L):
            candidates_all.append(subset)
    candidates = [list(filter(lambda x:m not in x, candidates_all)) for m in cand]
    
    if(os.path.exists(os.path.join(args.checkpoints,args.identifier))):
        if accelerator.is_main_process:
            with open(os.path.join(args.checkpoints, args.identifier, "wandb_run_id.txt"), "r") as f:
                run_id = f.read().strip()
                wandb.init(
                dir = os.path.join(os.path.expanduser("~"), ".wandb_logs"),
                project="HF_STUFF",
                id=run_id,
                resume="allow",
                config=vars(args)
                ) 
    else:
        if accelerator.is_main_process:
            os.makedirs(os.path.join(args.checkpoints, args.identifier), exist_ok=True)
            run = wandb.init(
            dir = os.path.join(os.path.expanduser("~"), ".wandb_logs"),
            project='HF_STUFF',
            name=args.identifier,
            config=vars(args)
            )
            try:
                with open(os.path.join(args.checkpoints, args.identifier, "wandb_run_id.txt"), "w") as f:
                    f.write(run.id)
            except:
                pass
    
    print('Training start',flush = True)
    while epoch < epochs:
        epoch +=1
        avg_train_total_loss = []
        avg_train_adversarial_loss = []
        avg_train_auxiliary_loss = []
        avg_train_d_real_loss = []
        avg_train_d_fake_loss = []
        
        for n, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
            if (n + 1) % args.save_freq == 0:
                #accelerator.save_state(output_dir=os.path.join(args.checkpoints,args.identifier))
                pass

            with accelerator.accumulate(generator):
                generator.train()
                discriminator.train()
                
                inputs_all = batch['image']# BxCxWxH
                targets = batch['target'] #Bx1xWxH
                modalities = batch['modality'].squeeze(dim=-1)
                segmentations = batch['segmentations'] #Bx1xWxH
                views = batch['views'].squeeze(dim=-1) 
                for m_shift in shift: # Each shift the missing modality is changed (shifted), making the model look at all the available data
                    if tversky_coeff != 0:
                        tversky_inputs_all = inputs_all.clone()
                    iterations += 1
                    if m_shift:
                        for idx in range(inputs_all.shape[0]):
                            modalities[idx] = modalities[idx]+1 if modalities[idx]!=3 else 0
                            targets[idx,:] = inputs_all[idx,modalities[idx]:modalities[idx]+1,:]

                    # ..._second --> something that will be used for the cycle loss
                    targets_second = torch.zeros_like(targets, device=targets.device)

                    inputs_masked = -1*torch.ones_like(inputs_all, device=targets.device)
                    inputs_masked_second = -1*torch.ones_like(inputs_all, device=targets.device)

                    modalities_second = torch.zeros_like(modalities, device=modalities.device)

                    # Lists of lists of the available modalities for the encoder. Ex: [ [1,2,3] , [0,2]]
                    input_modals = []
                    input_modals_second = [] # This one is for the cycle loss stuff

                    # Obscure code that completes the inputs in some manner
                    for n, m in enumerate(modalities): # For each missing modality given by the DataLoader
                        
                        if always_three:
                            cand = [x for x in [0,1,2,3] if x != m] # Three available modalities
                            input_modals.append(cand)
                            for c in cand:
                                inputs_masked[n,c,:] = inputs_all[n,c,:] # Unmask only the avaliable modalities
                                
                            # Between the available modalities, chooose another modality that will be missing on the second forward
                            m_masked = random.choice(cand)
                            modalities_second[n] = m_masked
                                
                            # On the second forward the available modalities will be all modals != missing_modality
                            cand_second = [x for x in [0,1,2,3] if x != m_masked]
                            input_modals_second.append(cand_second)
                            for c in cand_second:
                                inputs_masked_second[n,c,:] = inputs_all[n,c,:] 
                            targets_second[n,:] = inputs_all[n,m_masked:m_masked+1,:] # Setting the target for the cycle stuff

                            
                        else:
                            if n < inputs_all.shape[0] // 2: # In the first half of the batch the missing modality will be generated using 2 or 3 avail modalities
                                cand = random.choice(candidates[m]) # Choose a combination of available modalities (m will alway be absent)

                                input_modals.append(cand) # Add this modality to the list of available modalities
                                for c in cand:
                                    inputs_masked[n,c,:] = inputs_all[n,c,:] # Unmask only the avaliable modalities
                                    
                                # Between the available modalities, chooose another modality that will be missing on the second forward
                                m_masked = random.choice(cand)
                                modalities_second[n] = m_masked

                                # On the second forward the available modalities will be all modals != missing_modality
                                cand_second = [x for x in [0,1,2,3] if x != m_masked]
                                input_modals_second.append(cand_second)
                                for c in cand_second:
                                    inputs_masked_second[n,c,:] = inputs_all[n,c,:] 
                                targets_second[n,:] = inputs_all[n,m_masked:m_masked+1,:] # Unmasking of the available modalities
                                
                            else: # In the remaining half of the batch only one modality will be available
                                cand = random.choice([x for x in [0,1,2,3] if x != m]) # Choose one available modality
                                input_modals.append([cand]) 
                                inputs_masked[n,cand,:] = inputs_all[n,cand,:] # Unmasking
                                
                                # The missing modality from the first pass will be the available modality in the second
                                modalities_second[n] = cand 
                                
                                # The available modality in the second pass will be the missing modality from the first
                                cand_second = m
                                input_modals_second.append([cand_second])
                                inputs_masked_second[n,cand_second,:] = inputs_all[n,cand_second,:]
                                targets_second[n,:] = inputs_all[n,cand:cand+1,:]

                    background_mask = None
                    if mask_background:
                        background_mask = torch.zeros_like(targets,dtype=torch.bool) # (B,1,240,240)
                        for elem in range(inputs_all.shape[0]):
                            avail = input_modals[elem]
                            list_of_masks = [(inputs_all[elem,avail_mod] < mins[avail_mod]).int() for avail_mod in avail]
                            background_mask[elem,0] = sum(list_of_masks, torch.zeros_like(list_of_masks[0], dtype=torch.int)) >= 2
                    # Train the Generator
                    optimizer.zero_grad()
                    with accelerator.autocast():
                        # First forward
                        targets_recon, f_recon = generator(inputs_masked,input_modals,modalities,views=views)
                        
                        if mask_background:
                            targets_recon[background_mask] = -1

                        if tversky_coeff != 0:
                            for batch_index in range(modalities.shape[0]):
                                tversky_inputs_all[batch_index:batch_index+1,modalities[batch_index]:modalities[batch_index]+1] = targets_recon[batch_index:batch_index+1]
                            
                            recon_segmentations = segmenter(tversky_inputs_all)
                            tverskys, _, _ = tversky_loss(recon_segmentations,segmentations)

                            tversky = tverskys * tversky_coeff
                        else:
                            tversky = torch.tensor(0.0, device=inputs_masked.device)

                        # Reconstruction loss term
                        recon_loss = (torch.abs(targets - targets_recon)).mean()

                        # Plotting per-modality reconstruction losses
                        with torch.no_grad():
                            per_modality_losses = {}
                            for m in [0, 1, 2, 3]:
                                mask = (modalities == m)
                                if mask.any():
                                    loss_m = (torch.abs(targets[mask] - targets_recon[mask])).mean()
                                    per_modality_losses[f"train/recon_loss_modality_{m}"] = loss_m.item()
                            if accelerator.is_main_process:
                                wandb.log(per_modality_losses)

                        for n, m in enumerate(modalities):
                            inputs_masked_second[n,m,:] = targets_recon[n,0,:]

                        # Cycle loss part
                        if not skip_cycle:
                            targets_cycle, f_cycle = generator(inputs_masked_second,input_modals_second,modalities_second,views=views)
                            if mask_background:
                                targets_cycle[background_mask] = -1
                            feature_l1_loss = 1 - F.cosine_similarity(f_recon.flatten(1,-1), f_cycle.flatten(1,-1)).mean()
                            cycle_loss = (torch.abs(targets_second - targets_cycle)).mean()
                        else:
                            feature_l1_loss = torch.tensor(0,device=inputs_masked_second.device)
                            cycle_loss = torch.tensor(0,device=inputs_masked_second.device)
                            
                        # Adversary loss part
                        discriminator.eval()
                        logits_fake, labels_fake = discriminator(targets_recon)
                        discriminator.train()

                        valid=torch.ones(logits_fake.shape,device=accelerator.device)
                        fake=torch.zeros(logits_fake.shape,device=accelerator.device)
                        
                        # The closer is the discriminator's output to "real!", the better
                        adversarial_loss = loss_adversarial(logits_fake, valid)

                        # Classification loss provided by the discriminator
                        auxiliary_loss = loss_auxiliary(labels_fake, modalities)
                        
                        targets_zero_background = torch.zeros_like(targets)
                        recon_zero_background = torch.zeros_like(targets_recon)
                        brain_masks_list = []
                        
                        # SSIM loss part
                        for brain_slice in range(targets.shape[0]): # For each index in the batch
                            brain_mask = targets[brain_slice:brain_slice+1] > mins[modalities[brain_slice]]
                            brain_masks_list.append(brain_mask)
                            targets_zero_background[brain_slice:brain_slice+1] = targets[brain_slice:brain_slice+1]*brain_mask
                            recon_zero_background[brain_slice:brain_slice+1] = targets_recon[brain_slice:brain_slice+1]*brain_mask
                            
                        ssim_image = loss_ssim(targets_zero_background,recon_zero_background) 
                        brain_masks = torch.cat(brain_masks_list,dim=0)
                        
                        tumor_masks = segmentations!=0
                        healthy_masks = brain_masks & (~tumor_masks)
                        healthy_ssim = (ssim_image * healthy_masks).sum() / (healthy_masks.sum() + 1e3)
                        tumor_ssim = (ssim_image * tumor_masks).sum() / (tumor_masks.sum() + 1e-3)
                        
                        if ssim_mode == 'whole' and ssim_coeff != 0:
                            whole_ssim = (ssim_image * brain_masks).sum() / (brain_masks.sum() + 1e-3)
                            ssim_loss_term = ssim_coeff * (1-whole_ssim)
                            
                        elif ssim_mode == 'double' and ssim_coeff != 0: 
                            ssim_loss_term = ssim_coeff * ((1-healthy_ssim) + (1-tumor_ssim))
                        else: # if ssim_coeff == 0
                            ssim_loss_term = torch.tensor(0.0, device=ssim_image.device)
                            
                        total_loss = 10*recon_loss +0.25*adversarial_loss + 0.25*auxiliary_loss + 1*feature_l1_loss + 1*cycle_loss + ssim_loss_term + tversky
                        
                        if accelerator.is_main_process:
                            wandb.log({
                                'train/running_g_loss': total_loss.item(),
                                'train/tumor_ssim': tumor_ssim.item(),
                                'train/healthy_ssim':healthy_ssim.item()
                            })
                    
                    # Out of the autocast section, backward pass
                    accelerator.backward(total_loss)

                    # Gradient clipping for stability
                    clip_grad_norm_(generator.parameters(), max_norm=0.1)

                    optimizer.step()

                    # Train the Discriminator
                    optimizer_D.zero_grad()
                    with accelerator.autocast():
                        # Forward pass
                        logits_real, labels_real = discriminator(targets_second)
                        logits_fake, labels_fake = discriminator(targets_recon.detach())
                        
                        # Learning to discriminate
                        d_real_adv = loss_adversarial(logits_real, valid)
                        d_fake_adv = loss_adversarial(logits_fake, fake)
                        
                        # Leaning to classify
                        d_real_aux = loss_auxiliary(labels_real, modalities_second)
                        d_fake_aux = loss_auxiliary(labels_fake, modalities)

                        d_loss = 0.25*(d_real_adv + d_fake_adv) + 0.25*d_real_aux + 0.25*d_fake_aux
                    # Out of the autocast section, backward pass
                    accelerator.backward(d_loss)

                    # Gradient clipping for stability
                    clip_grad_norm_(discriminator.parameters(), max_norm=0.1)
                    optimizer_D.step()

                    scheduler.step(iterations)
                    scheduler_D.step(iterations)

                    if accelerator.is_main_process:
                        wandb.log({
                            'train/running_d_loss': d_loss.item()
                        })
                    avg_train_total_loss.append(total_loss.item())
                    avg_train_adversarial_loss.append(adversarial_loss.item())
                    avg_train_auxiliary_loss.append(auxiliary_loss.item())
                    avg_train_d_real_loss.append(d_real_adv.item())
                    avg_train_d_fake_loss.append(d_fake_adv.item())
                    if accelerator.is_main_process:
                        try:
                            wandb.log({
                                'lr/lr_G': optimizer.param_groups[0]['lr'],
                                'lr/lr_D': optimizer_D.param_groups[0]['lr']
                            })
                        except:
                            print('Failed to log learning rates')
                    
                  
                    

        if accelerator.is_main_process:
            print(f"Train Loss: {np.mean(avg_train_total_loss):.6f}, G Loss: {np.mean(avg_train_adversarial_loss)+np.mean(avg_train_auxiliary_loss):.6f}, D Loss: {np.mean(avg_train_d_real_loss)+np.mean(avg_train_d_fake_loss):.6f}")
            wandb.log({
                'train/train_loss': np.mean(avg_train_total_loss),
                'train/g_loss': np.mean(avg_train_adversarial_loss)+np.mean(avg_train_auxiliary_loss),
                'train/d_loss':np.mean(avg_train_d_real_loss)+np.mean(avg_train_d_fake_loss)
            })

        if epoch % valid_epochs == 0:
            # Saving at the end of the evaluation
            accelerator.save_state(output_dir=os.path.join(args.checkpoints, args.identifier))
            with torch.no_grad():
                avg_valid_recon_loss = []
                avg_valid_psnr = []
                avg_valid_ssim = []
                avg_valid_msssim = []
                tumor_ssims = []
                healthy_ssims = []
                dice_scores = {
                    label: [] for label in range (1,4)
                }
                for batch in valid_loader:
                    generator.eval()
                    
                    inputs = batch['image_masked'] # BxCxWxH
                    targets = batch['target']
                    modalities = batch['modality'].squeeze(dim=-1)
                    segmentations = batch['segmentations']
                    views = batch['views'].squeeze(dim=-1)
                    swap_segmentations(segmentations)

                    input_modals = []
                    for n, m in enumerate(modalities):
                        input_modals.append([x for x in [0,1,2,3] if x != m])
                    targets_recon, _ = generator(inputs,input_modals,modalities,train_mode=False,views = views)


                    if mask_background:
                        background_mask = torch.zeros_like(targets,dtype=torch.bool) # (B,1,240,240)
                        for elem in range(inputs.shape[0]):
                            avail = input_modals[elem]
                            list_of_masks = [(inputs[elem,avail_mod] < mins[avail_mod]).int() for avail_mod in avail]
                            background_mask[elem,0] = sum(list_of_masks, torch.zeros_like(list_of_masks[0], dtype=torch.int)) >= 2
                        targets_recon[background_mask]=-1
                    
                    # Validation metrics
                    for j in range(targets_recon.shape[0]):
                        avg_valid_recon_loss.append(torch.abs(targets[j:j+1,:] - targets_recon[j:j+1,:]).mean().cpu().detach())
                        avg_valid_psnr.append(metric_psnr(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu().detach())
                        avg_valid_ssim.append(metric_siim(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu().detach())
                        avg_valid_msssim.append(metric_mssiim(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu().detach())

                    # DICE part
                    if tversky_coeff != 0:
                        dice_inputs_val = inputs.clone()
                        for modal_index in range(modalities.size(0)):
                            dice_inputs_val[modal_index,modalities[modal_index]:modalities[modal_index]+1] = targets_recon[modal_index]
                        val_segmentations = torch.argmax(segmenter(dice_inputs_val),dim=1)
                        dice_dict = per_label_dice_coefficient(val_segmentations,segmentations.squeeze(1))
                        for index,value in dice_dict.items():
                            if value >= 0 and index !=0:
                                dice_scores[index].append(value)
                    # Validation SSIM
                    brain_masks_list = []
                    targets_zero_background = torch.zeros_like(targets)
                    recon_zero_background = torch.zeros_like(targets_recon)

                    for brain_slice in range(targets.shape[0]):
                        brain_mask = targets[brain_slice:brain_slice+1] > mins[modalities[brain_slice]]
                        brain_masks_list.append(brain_mask)
                        targets_zero_background[brain_slice:brain_slice+1] = targets[brain_slice:brain_slice+1]*brain_mask
                        recon_zero_background[brain_slice:brain_slice+1] = targets_recon[brain_slice:brain_slice+1]*brain_mask
                    ssim_image = loss_ssim(targets_zero_background,recon_zero_background)
                    brain_masks = torch.cat(brain_masks_list,dim=0)
                    tumor_masks = segmentations!=0
                    healthy_masks = brain_masks & (~tumor_masks)
                    healthy_ssim = ssim_image[healthy_masks].mean() if ssim_image[healthy_masks].numel() > 0 else None
                    tumor_ssim = ssim_image[tumor_masks].mean() if ssim_image[tumor_masks].numel() > 0 else None

                    if healthy_ssim is not None:
                        healthy_ssims.append(healthy_ssim)
                    if tumor_ssim is not None:
                        tumor_ssims.append(tumor_ssim)
                
                if accelerator.is_main_process and tversky_coeff != 0:
                    wandb.log({
                        f'val/dice_{index}': np.mean(value) for index, value in dice_scores.items() 
                    })
                if accelerator.is_main_process:
                    #print(f"Valid Recon Loss: {np.mean(avg_valid_recon_loss):.6f}, PSNR: {np.mean(avg_valid_psnr):.6f}, , SSIM: {np.mean(avg_valid_ssim):.6f}, , MS-SSIM: {np.mean(avg_valid_msssim):.6f}")
                    wandb.log({
                        'val/recon_loss': np.mean(avg_valid_recon_loss),
                        'val/psnr':np.mean(avg_valid_psnr),
                        'val/ssim': np.mean(avg_valid_ssim),
                        'val/ms_ssim': np.mean(avg_valid_msssim),
                        'val/healthy_ssim':torch.mean(torch.stack(healthy_ssims)) if len(healthy_ssims) > 0 else torch.tensor(0.0),
                        'val/tumor_ssim':torch.mean(torch.stack(tumor_ssims))if len(tumor_ssims) > 0 else torch.tensor(0.0)
                    })

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            epoch_state_path = os.path.join(args.checkpoints, args.identifier,"current_epoch.pth")
            epoch_state = {'epoch': epoch}
            torch.save(epoch_state, epoch_state_path)

        '''if accelerator.is_main_process:
            print(f'saved the state of the model for epoch-->{epoch}')
            fig_rf = get_real_fake_comparison_graph(epoch, 0, args.identifier)
            wandb.log({"images/real_fake_comparison": wandb.Image(fig_rf)})
            plt.close(fig_rf)'''
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
    
