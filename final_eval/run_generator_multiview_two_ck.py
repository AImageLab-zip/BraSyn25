import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.hf_gan import HFGAN_3D
from src.data.dataset import BrainDataset3D,ULTIMATE_brain_coll_3d
from pathlib import Path
import shutil
import os
from tqdm import tqdm
import nibabel as nib
from src.utils.saving import save_tensor_to_nii
from argparse import ArgumentParser

import numpy as np
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  

parser = ArgumentParser()
parser.add_argument('--id',type = str,required=True,help='Provide the base identifier for the run')
parser.add_argument('--id-axi',type = str,required=True,help='Provide the base identifier for the axial checkpoint')
parser.add_argument('--id-sag',type = str,required=True,help='Provide the base identifier for the saggital checkpoint')

args = parser.parse_args()
id = args.id
id_axi = args.id_axi
id_sag = args.id_sag

# Remember that only checkpoints available in the "final" directory will be usable 
weights_file_axi = Path(f'/work_path/work_path/checkpoints/final/{id_axi}.safetensors')
weights_file_sag = Path(f'/work_path/work_path/checkpoints/final/{id_sag}.safetensors')
# Original and immutable val dataset
input_path = Path(f'/work_path/work_path/brats3d/pseudo_random/original')

# These paths will contain the 3 original input modals + the reconstructed modal
output_path_gli_complete = Path(f'/work_path/brats3d/pseudo_random/complete_recon_gli_{id}') # nnunet convention
output_path_met_complete = Path(f'/work_path/brats3d/pseudo_random/complete_recon_met_{id}') # brats convention

# These paths will only contain the reconstructed modality, in nnunet_convention
output_path_gli = Path(f'/work_path/brats3d/pseudo_random/recon_gli_{id}')
output_path_met = Path(f'/work_path/brats3d/pseudo_random/recon_met_{id}')


infuse_view = False
grouped_encoder = False

# Minor parameters for the model
if '24' in id or '25' in id:
    grouped_encoder = True
if '25' in id:
    infuse_view = True

# Hardcoded pre-computed scores for the gli + met dataset
means=torch.tensor([1066.3375,  781.2247,  510.9852,  673.4393])
stds = torch.tensor([1301.7006,  944.3418,  769.4159,  804.3864])
max_vals=torch.tensor([8664, 7315, 8842, 8233])
mins = (torch.zeros_like(means) - means)/stds

# These names correspond to indexes:[  0   ,   1  ,  2   .  3   ] in the tensors
modal_name_list_brats =             ['t1c' ,'t1n' ,'t2f' ,'t2w' ]
modal_name_list_nnunet =            ['0001','0002','0003','0000']

# This collate should be compatible with everything
dataset = BrainDataset3D(dataset_dirs=input_path)
dataloader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle = False,
                        num_workers=4,
                        collate_fn=ULTIMATE_brain_coll_3d)
assert dataloader.batch_size == 1, 'KEEP THE BATCH SIZE OF THE DATALOADER TO 1 PLEASE DO NOT CHANGE THE BATCH SIZE PLSPLS'

if os.path.exists(output_path_gli_complete):
    shutil.rmtree(output_path_gli_complete)
os.makedirs(output_path_gli_complete)

if os.path.exists(output_path_met_complete):
    shutil.rmtree(output_path_met_complete)
os.makedirs(output_path_met_complete)

if os.path.exists(output_path_gli):
    shutil.rmtree(output_path_gli)
os.makedirs(output_path_gli)

if os.path.exists(output_path_met):
    shutil.rmtree(output_path_met)
os.makedirs(output_path_met)

# Our cutting edge technology, super efficient, so wise and powerful, wrapper model (it sucks)
model_axi= HFGAN_3D(view='axi',weights_file=weights_file_axi,infuse_view=infuse_view,grouped_encoder=grouped_encoder)
mode_sag = HFGAN_3D(view='sag',weights_file=weights_file_sag,infuse_view=infuse_view,grouped_encoder=grouped_encoder)
# Dataset iteration with conditional treatment for mets and glis
for element in tqdm(dataloader,total=len(dataloader)):
    brains = element['images']
    missing_modalities = element['missing_modalities']
    recon_axi, everything_axi = model_axi(brains,missing_modalities)
    recon_sag, everything_sag = model_axi(brains,missing_modalities)

    recon , everything = (recon_axi + recon_sag)/2 , (everything_axi + everything_sag)/2
    headers = element['image_headers'][0]
    affines = element['image_affines'][0]
    # De-normalizations
    everything = (everything * stds.view(1, 4, 1, 1, 1)) + means.view(1, 4, 1, 1, 1)
    everything[everything<0] = 0

    recon = (recon * stds[missing_modalities.item()]) + means[missing_modalities.item()]
    recon[recon<0] = 0

    # The mets are saved with the brats convention
    if 'MET' in str(Path(element['ids'][0]).name):
        output_path = output_path_met_complete
        os.makedirs(output_path / Path(element['ids'][0]).name)
        for i in range(4):
            save_path = output_path / Path(element['ids'][0]).name / (str(Path(element['ids'][0]).name)+ '-' + modal_name_list_brats[i] + '.nii.gz')
            save_tensor_to_nii(everything[:,i:i+1].squeeze(0),save_path,affines[i],headers[i])

            # The single reconstructed modalities are saved 
            if i == missing_modalities.item():
                save_path = output_path_met / (str(Path(element['ids'][0]).name)+ '_' + modal_name_list_nnunet[i] + '.nii.gz')
                save_tensor_to_nii(recon.squeeze(0),save_path,affines[i],headers[i])
    # The glis are saved with the nnunet convention
    elif 'GLI' in str(Path(element['ids'][0]).name):
        output_path = output_path_gli_complete
        for i in range(4):
            save_path = output_path / (str(Path(element['ids'][0]).name)+ '_' + modal_name_list_nnunet[i] + '.nii.gz')
            save_tensor_to_nii(everything[:,i:i+1].squeeze(0),save_path,affines[i],headers[i])
            if i == missing_modalities.item():
                save_path = output_path_gli / (str(Path(element['ids'][0]).name)+ '_' + modal_name_list_nnunet[i] + '.nii.gz')
                save_tensor_to_nii(recon.squeeze(0),save_path,affines[i],headers[i])
    else:
        raise RuntimeError('Invalid type of tumor!')
    

print('End of the generation phase!')
    
