import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import HFGAN_3D
from utils import BrainDataset3D, ULTIMATE_brain_coll_3d
from pathlib import Path
import shutil
import os
import nibabel as nib
from utils import save_tensor_to_nii
import numpy as np
import random
from tqdm import tqdm

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

view = os.environ['VIEW']

input_path = Path('/input')
output_path = Path('/output')

# Remember that only checkpoints available in the "final" directory will be usable
weights_file = Path('checkpoint/model.safetensors')

infuse_view = (os.environ['INFUSE_VIEW'].lower() == 'true')
grouped_encoder = (os.environ['GRP_ENCODER'].lower() == 'true')

# Hardcoded pre-computed scores for the gli + met dataset
means = torch.tensor([1066.3375, 781.2247, 510.9852, 673.4393])
stds = torch.tensor([1301.7006, 944.3418, 769.4159, 804.3864])
max_vals = torch.tensor([8664, 7315, 8842, 8233])
mins = (torch.zeros_like(means) - means) / stds

# These names correspond to indexes:[  0   ,   1  ,  2   .  3   ] in the tensors
modal_name_list_brats = ['t1c', 't1n', 't2f', 't2w']
modal_name_list_nnunet = ['0001', '0002', '0003', '0000']

# This collate should be compatible with everything
dataset = BrainDataset3D(dataset_dirs=input_path)
dataloader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=1,
                        collate_fn=ULTIMATE_brain_coll_3d)
assert dataloader.batch_size == 1, 'KEEP THE BATCH SIZE OF THE DATALOADER TO 1 PLEASE DO NOT CHANGE THE BATCH SIZE PLSPLS'

if os.path.exists(output_path):
    for file_path in output_path.iterdir():
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# Our cutting edge technology, super efficient, so wise and powerful, wrapper model (it sucks)
model = HFGAN_3D(view=view, weights_file=weights_file, infuse_view=infuse_view, grouped_encoder=grouped_encoder)

# Dataset iteration with conditional treatment for mets and glis
for element in tqdm(dataloader, total=len(dataloader)):
    brains = element['images']
    missing_modalities = element['missing_modalities']
    recon, _ = model(brains, missing_modalities)

    headers = element['image_headers'][0]
    affines = element['image_affines'][0]

    # De-normalizations
    recon = (recon * stds[missing_modalities.item()]) + means[missing_modalities.item()]
    recon[recon < 0] = 0

    i = missing_modalities.item()
    save_path = output_path / (
            str(Path(element['ids'][0]).name) + '-' + modal_name_list_brats[i] + '.nii.gz')
    save_tensor_to_nii(recon.squeeze(0), save_path, affines[i], headers[i])

print('End of the generation phase!')

