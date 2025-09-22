## This script creates a pseudo validation set

import os
import random 
import numpy as np
import shutil

# The target validation set foler and the new folder, please replace it with yours
val_set_folders = ['input_folder/val_gli', 'input_folder/val_met'] # Put the downloaded validation folders here
val_set_missing = 'output_folder/pseudo_random' # Put your output_folder here
shutil.rmtree(val_set_missing)
if not os.path.exists(val_set_missing):
    os.mkdir(val_set_missing)

# create a pseudo validation set by randomly dropping one modality
# This is official BraTS code adapted with a file copy
np.random.seed(123456)  # Fix random seed
modality_list = ['t1c', 't1n', 't2f', 't2w']  # The list of modalities in the given folder
for val_set_folder in val_set_folders:
    folder_list = os.listdir(val_set_folder)
    folder_list.sort()
    drop_index = np.random.randint(0, 4, size=len(folder_list))


    for count, ff in enumerate(folder_list):
        if not os.path.exists(os.path.join(val_set_missing, ff)):
            os.mkdir(os.path.join(val_set_missing, ff))
        
        file_list = os.listdir(os.path.join(val_set_folder, ff))

        for mm in file_list:
            print(modality_list[drop_index[count]] + ' is droppd for case ' + mm)
            if not modality_list[drop_index[count]] in mm:
                shutil.copyfile(os.path.join(val_set_folder, ff, mm), os.path.join(val_set_missing, ff, mm))