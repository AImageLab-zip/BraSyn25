# No More Slice Wars: Just code
This repository contains the code and configuration files used for our submission to the 
Brain Tumor Segmentation Challenge (BraTS 2025), Task 8.
## 0. System requirements
This code was tested on ubuntu 22.04 LTS and CUDA 12.6, but will work on Windows too. 
## 1. Repository Structure

```
BraSyn25/
├── final_eval/ # Folder for the final local evaluation
│   └── sbatchable_scripts/     # Sbatch files specific for evaluation
├── slurm/                      # Examples of sbatch files used during training
│   ├── sbatchable_scripts/
│   └── segmenter_script/
├── src/
│   ├── data/                   # Dataset code, collate functions and stuff
│   ├── models/                 # Pytorch models
│   ├── training/               # Training scripts
│   └── utils/                  # Utilities
└── README.md                   # This file! :)
```

## 2. Data Preparation
This year's challenge used 3 different datasets:
- Glioma.
- Metastasis.
- Metastasis - add (additional samples).

After downloading, prepare the 3d original dataset following this structure:
```
work_path/
└── brats3d/
    ├── train_gli/                  
    ├── tran_met/                 
    ├── train_met_add/              
    ├── val_gli/              
    └── val_met/                 
```
Remember to edit the file paths in the code you are using!
In the validation dataset, segmentations are missing. We obtained good quality segmentations, 
used only during the training phase, using nnunet. You can obtain them using the newly released 
brats package too. Make sure to get those segmentations before jumping to the next step.
After downloading the dataset, you will have to slice it, running
```src/data/dataset_slicer.py```. \
This will create the 2d dataset directories, following the sctructure:

```
work_path/
├── brats2d_axi/
│   ├── train_gli/                   
│   ├── tran_met/                    
│   ├── train_met_add/               
│   ├── val_gli/             
│   └── val_met/     
├── brats2d_cor/
│   ├── train_gli/                   
│   ├── tran_met/                    
│   ├── train_met_add/               
│   ├── val_gli/             
│   └── val_met/  
├── brats2d_sag/
│   ├── train_gli/                   
│   ├── tran_met/                    
│   ├── train_met_add/               
│   ├── val_gli/             
│   └── val_met/  
└── brats3d/
    ├── train_gli/                  
    ├── tran_met/                 
    ├── train_met_add/              
    ├── val_gli/              
    └── val_met/ 
```
## 3. Validation
To get the segmentation scores on the official validation dataset, you will have to obtain 
segmentation pseudo-labels first. You can reuse the ones from the data preparation step.
Put the obtained segmentations into ```/work_path/brats3d/pseudo_random/gli_segmentations``` and 
```/work_path/brats3d/pseudo_random/gli_segmentations```. Remember to replace ```/work_path``` with your
actual work directory in the source code. For the segmentation part of the 3d validation phase, 
since it is from the brats python package, a cuda capable gpu is mandatory.

## 4. Training
If you bravely want to train the model, the available code uses pytorch and accelerate and using
cuda capable gpus is mandatory (since training it on cpu would take ages). Please, check the cuda compute
capability of your system before using advanced features such as AMP, model compilation and flash attention. \
link --> https://developer.nvidia.com/cuda-gpus 

### Wandb
Wandb is included in the training script. To make it work, make sure to log it into
wandb from you terminal.

### Resources
The model was trained on 2/4 NVIDIA L40S (48GB each). 
Adjust your settings and your batch size accordingly.

## 5 Checkpoints
You can find the best checkpoints at: https://drive.google.com/drive/folders/1u7mJN9BZ-a5HLz8IMDrU0T1P3UEYGGe7?usp=sharing
Initialize your model like:
```model = HFGAN(dim=64, num_inputs=4, num_outputs=1, dim_mults=(1,2,4,8,10), n_layers=4, skip=True, blocks=False,grouped_encoder=grouped_encoder, infuse_view=False)```
Where:
''grouped_encoder == True''' if you want to run a checkpoint from run24, '''False''' otherwhise.

## 6. Issues?
Feel free to contact me at: ```269868@studenti.unimore.it``` if you got any issues
running the code.
