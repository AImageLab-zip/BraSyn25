from argparse import ArgumentParser
from pathlib import Path
from src.utils.preprocessing import get_tensor_header_affine_from_nii
import numpy as np
import torch
from typing import cast
import nibabel as nib

# MONAI imports
from monai.metrics.meandice import compute_dice
from monai.metrics import compute_surface_dice

def process_file(file, segm_reference_path_gli, segm_reference_path_met):
    """Process a single file"""
    if 'MET' in str(file):
        base_segm_path = segm_reference_path_met
        tumor_type = 'MET'
    elif 'GLI' in str(file):
        base_segm_path = segm_reference_path_gli
        tumor_type = 'GLI'
    else:
        raise RuntimeError('Invalid type of tumor!')
    
    ref_seg,_,_ = get_tensor_header_affine_from_nii(base_segm_path / file.name)
    seg, header, _ = get_tensor_header_affine_from_nii(file)
    header = cast(nib.Nifti1Header, header)
    
    # Convert to torch tensors for MONAI
    ref_seg = torch.from_numpy(np.asarray(ref_seg, dtype=np.uint8))
    seg = torch.from_numpy(np.asarray(seg, dtype=np.uint8))
    
    # Get spacing and convert to float64 for MONAI
    try:
        spacing = np.array(header.get_zooms()[:3], dtype=np.float64)
    except:
        spacing = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    
    # Calculate metrics for each class separately
    class_results = {}
    
    # Define class mappings
    classes = {
        'ET': {'pred': (seg == 1), 'gt': (ref_seg == 1)},
        'TC': {'pred': (seg == 3) | (seg == 2), 'gt': (ref_seg == 3) | (ref_seg == 2)},
        'WT': {'pred': (seg >= 1) & (seg <= 3), 'gt': (ref_seg >= 1) & (ref_seg <= 3)}
    }
    
    for class_name, masks in classes.items():
        pred_mask = masks['pred'].unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W, D]
        gt_mask = masks['gt'].unsqueeze(0).unsqueeze(0).float()      # [1, 1, H, W, D]
        
        # Calculate Dice score for this class
        try:
            dice_score = compute_dice(
                y_pred=pred_mask,
                y=gt_mask,
                include_background=False,
                ignore_empty=True
            )
            dice_value = float(dice_score.squeeze().numpy())
            # Handle NaN for empty cases
            if np.isnan(dice_value):
                dice_value = 1.0 if (pred_mask.sum() == 0 and gt_mask.sum() == 0) else 0.0
        except Exception as e:
            print(f"Dice calculation failed for {file.name} - {class_name}: {e}")
            dice_value = 0.0
        
        # Calculate NSD score for this class
        try:
            nsd_score = compute_surface_dice(
                y_pred=pred_mask,
                y=gt_mask,
                class_thresholds=[1.0],  # 1mm threshold
                include_background=False,
                spacing=spacing
            )
            nsd_value = float(nsd_score.squeeze().numpy())
            # Handle NaN for empty cases
            if np.isnan(nsd_value):
                nsd_value = 1.0 if (pred_mask.sum() == 0 and gt_mask.sum() == 0) else 0.0
        except Exception as e:
            print(f"NSD calculation failed for {file.name} - {class_name}: {e}")
            nsd_value = 0.0
        
        class_results[f'{class_name}_Dice'] = dice_value
        class_results[f'{class_name}_NSD'] = nsd_value
        
        print(f"{file.name} - {class_name}: Dice = {dice_value:.4f}, NSD = {nsd_value:.4f}")
    
    return file.name, tumor_type, class_results

def main():
    parser = ArgumentParser()
    parser.add_argument('--id', type=str, required=True, help='Provide the base identifier of the checkpoint.')
    args = parser.parse_args()
    id = args.id

    segm_reference_path_gli = Path('/work_path/brats3d/pseudo_random/gli_segmentations')
    segm_reference_path_met = Path('/work_path/brats3d/pseudo_random/met_segmentations')

    segm_path_gli = Path(f'/work_path/brats3d/pseudo_random/gli_segm_{id}')
    segm_path_met = Path(f'/work_path/brats3d/pseudo_random/met_segm_{id}')

    file_list = list(segm_path_gli.iterdir()) + list(segm_path_met.iterdir())

    # Initialize results
    gli_results = {'ET_Dice': [], 'TC_Dice': [], 'WT_Dice': [], 'ET_NSD': [], 'TC_NSD': [], 'WT_NSD': []}
    met_results = {'ET_Dice': [], 'TC_Dice': [], 'WT_Dice': [], 'ET_NSD': [], 'TC_NSD': [], 'WT_NSD': []}
    all_results = {'ET_Dice': [], 'TC_Dice': [], 'WT_Dice': [], 'ET_NSD': [], 'TC_NSD': [], 'WT_NSD': []}

    # Process files sequentially
    for file in file_list:
        filename, tumor_type, file_results = process_file(
            file, segm_reference_path_gli, segm_reference_path_met
        )
        
        current_results = gli_results if tumor_type == 'GLI' else met_results
        
        for metric in ['ET_Dice', 'TC_Dice', 'WT_Dice', 'ET_NSD', 'TC_NSD', 'WT_NSD']:
            value = file_results[metric]
            current_results[metric].append(value)
            all_results[metric].append(value)

    # Calculate means using numpy for speed
    all_dice_means = [np.mean(all_results['ET_Dice']), np.mean(all_results['TC_Dice']), np.mean(all_results['WT_Dice'])]
    all_nsd_means = [np.mean(all_results['ET_NSD']), np.mean(all_results['TC_NSD']), np.mean(all_results['WT_NSD'])]

    gli_dice_means = [np.mean(gli_results['ET_Dice']), np.mean(gli_results['TC_Dice']), np.mean(gli_results['WT_Dice'])]
    gli_nsd_means = [np.mean(gli_results['ET_NSD']), np.mean(gli_results['TC_NSD']), np.mean(gli_results['WT_NSD'])]

    met_dice_means = [np.mean(met_results['ET_Dice']), np.mean(met_results['TC_Dice']), np.mean(met_results['WT_Dice'])]
    met_nsd_means = [np.mean(met_results['ET_NSD']), np.mean(met_results['TC_NSD']), np.mean(met_results['WT_NSD'])]

    # Write results
    out_file = f'/home/user/results/{id}_dice_nsd.txt'
    with open(out_file, 'w') as f:
        print(f'\nDice scores for id:{id}, in the order: ET, TC, WT', file=f)
        print(f'total_dice: {all_dice_means}', file=f)
        print(f'gli_dice: {gli_dice_means}', file=f)
        print(f'met_dice: {met_dice_means}', file=f)

        print(f'\nNSD scores for id:{id}, in the order: ET, TC, WT', file=f)
        print(f'total_nsd: {all_nsd_means}', file=f)
        print(f'gli_nsd: {gli_nsd_means}', file=f)
        print(f'met_nsd: {met_nsd_means}', file=f)

if __name__ == '__main__':
    main()