from pathlib import Path
from src.utils.indexing import index_to_nnunet_name
import shutil
import os
from tqdm import tqdm
brats_path = Path('/input/path')
nnunet_path = Path('/output/path')
if os.path.exists(nnunet_path):
    shutil.rmtree(nnunet_path)
os.makedirs(nnunet_path)
for folder in tqdm(brats_path.iterdir()):
    for file in folder.iterdir():
        if 't1c' in str(file):
            shutil.copy(file,nnunet_path/(str(folder.name)+'_0001.nii.gz'))
        elif 't1n' in str(file):
            shutil.copy(file,nnunet_path/(str(folder.name)+'_0002.nii.gz'))
        elif 't2f' in str(file):
            shutil.copy(file,nnunet_path/(str(folder.name)+'_0003.nii.gz'))
        elif 't2w' in str(file):
            shutil.copy(file,nnunet_path/(str(folder.name)+'_0000.nii.gz'))
        else:
            pass