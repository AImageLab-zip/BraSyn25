from argparse import ArgumentParser
from pathlib import Path
from src.utils.preprocessing import get_tensor_header_affine_from_nii
import numpy as np
from scipy import ndimage
from typing import cast
import nibabel as nib

parser = ArgumentParser()
parser.add_argument('--id',type = str,required=True,help='Provide the base identifier of the checkpoint.\n'+
                    'Ex: if the checkpoint is at /work_path/checkpoints/final/run_14_always_three.safetensors,\n' +
                    'the id should be: --id run_14_always_three')
args = parser.parse_args()
id = args.id

segm_reference_path_gli = Path('/work_path/brats3d/pseudo_random/gli_segmentations') # nnunet convention
segm_reference_path_met = Path('/work_path/brats3d/pseudo_random/met_segmentations') # nnunet convention

segm_path_gli = Path(f'/work_path/brats3d/pseudo_random/gli_segm_{id}') # nnunet convention
segm_path_met = Path(f'/work_path/brats3d/pseudo_random/met_segm_{id}') # nnunet convention

file_list = list(segm_path_gli.iterdir()) + list(segm_path_met.iterdir())

# Store results
gli_results = {'ET_Dice': [], 'TC_Dice': [], 'WT_Dice': [], 'ET_NSD': [], 'TC_NSD': [], 'WT_NSD': []}
met_results = {'ET_Dice': [], 'TC_Dice': [], 'WT_Dice': [], 'ET_NSD': [], 'TC_NSD': [], 'WT_NSD': []}
all_results = {'ET_Dice': [], 'TC_Dice': [], 'WT_Dice': [], 'ET_NSD': [], 'TC_NSD': [], 'WT_NSD': []}

for file in file_list:
    if 'MET' in str(file):
        base_segm_path = segm_reference_path_met
        current_results = met_results
    elif 'GLI' in str(file):
        base_segm_path = segm_reference_path_gli
        current_results = gli_results
    else:
        raise RuntimeError('Invalid type of tumor!')
    
    ref_seg,_,_ = get_tensor_header_affine_from_nii(base_segm_path / file.name)
    seg ,header,_= get_tensor_header_affine_from_nii(file)
    header = cast(nib.Nifti1Header, header)
    ref_seg = np.array(ref_seg)
    seg = np.array(seg)
    
    # Create aggregated labels
    classes = {
        'ET': [(seg == 3), (ref_seg == 3)],
        'TC': [(seg == 1) | (seg == 3), (ref_seg == 1) | (ref_seg == 3)],
        'WT': [(seg == 1) | (seg == 2) | (seg == 3), (ref_seg == 1) | (ref_seg == 2) | (ref_seg == 3)]
    }

    for class_name, (pred, gt) in classes.items():
        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)
        
        # Dice score
        if pred.sum() == 0 and gt.sum() == 0:
            print('Error')
            dice = 1.0
        else:
            dice = 2.0 * (pred * gt).sum() / (pred.sum() + gt.sum())
        
        # NSD score
        pred_surface = pred ^ ndimage.binary_erosion(pred)
        gt_surface = gt ^ ndimage.binary_erosion(gt)
        
        # Get voxel spacing
        try:
            spacing = np.array(header.get_zooms()[:3])
        except:
            spacing = np.array([1.0, 1.0, 1.0])
            print(f"Warning: Using unit spacing for {file.name}")
            
        if not np.any(pred_surface) or not np.any(gt_surface):
            print('Error!')
            nsd = 1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0
        else:
            # Calculate surface distances
            pred_coords = np.column_stack(np.where(pred_surface)) * spacing
            gt_coords = np.column_stack(np.where(gt_surface)) * spacing
            
            # Distances from pred to gt and gt to pred
            distances = []
            for pred_point in pred_coords:
                distances.append(np.min(np.sqrt(np.sum((gt_coords - pred_point) ** 2, axis=1))))
            for gt_point in gt_coords:
                distances.append(np.min(np.sqrt(np.sum((pred_coords - gt_point) ** 2, axis=1))))
            
            nsd = np.mean(np.array(distances) <= 1.0)
        
        
        # Store in appropriate result lists
        current_results[f'{class_name}_Dice'].append(dice)
        current_results[f'{class_name}_NSD'].append(nsd)
        all_results[f'{class_name}_Dice'].append(dice)
        all_results[f'{class_name}_NSD'].append(nsd)
        
        print(f"{file.name} - {class_name}: Dice = {dice:.4f}, NSD = {nsd:.4f}")


# Calculate means
all_dice_means = [np.mean(all_results['ET_Dice']), np.mean(all_results['TC_Dice']), np.mean(all_results['WT_Dice'])]
all_nsd_means = [np.mean(all_results['ET_NSD']), np.mean(all_results['TC_NSD']), np.mean(all_results['WT_NSD'])]

gli_dice_means = [np.mean(gli_results['ET_Dice']), np.mean(gli_results['TC_Dice']), np.mean(gli_results['WT_Dice'])]
gli_nsd_means = [np.mean(gli_results['ET_NSD']), np.mean(gli_results['TC_NSD']), np.mean(gli_results['WT_NSD'])]

met_dice_means = [np.mean(met_results['ET_Dice']), np.mean(met_results['TC_Dice']), np.mean(met_results['WT_Dice'])]
met_nsd_means = [np.mean(met_results['ET_NSD']), np.mean(met_results['TC_NSD']), np.mean(met_results['WT_NSD'])]

# Print results in the requested format
out_file = f'/home/user/results/results/{id}_dice_nsd.txt'
with open(out_file, 'w') as f:
    print(f'\nDice scores for id:{id}, in the order: ET, TC, WT',file=f)
    print(f'total_dice: {all_dice_means}',file=f)
    print(f'gli_dice: {gli_dice_means}',file=f)
    print(f'met_dice: {met_dice_means}',file=f)

    print(f'\nNSD scores for id:{id}, in the order: ET, TC, WT',file=f)
    print(f'total_nsd: {all_nsd_means}',file=f)
    print(f'gli_nsd: {gli_nsd_means}',file=f)
    print(f'met_nsd: {met_nsd_means}',file=f)