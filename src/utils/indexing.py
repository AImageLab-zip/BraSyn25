import nibabel as nib
import gzip
import torch
from pathlib import Path
import io
import numpy as np
from typing_extensions import deprecated

@deprecated("Using 0 for the segmentation is bad and confusing")
def name_to_index_DEPRECATED(name:str)->int:
    '''
    example of order 
    BraTS-GLI-01368-000-seg.nii.gz
    BraTS-GLI-01368-000-t1c.nii.gz
    BraTS-GLI-01368-000-t1n.nii.gz
    BraTS-GLI-01368-000-t2f.nii.gz
    BraTS-GLI-01368-000-t2w.nii.gz
    '''
    match name:
        case 'segmentation':
            return 0
        case 't1_contrast':
            return 1
        case 't1_normal':
            return 2
        case 't2_flair':
            return 3
        case 't2_weighted':
            return 4
        case _:
            raise ValueError("Slice names must be one of: \
                             'segmentation', 't1_contrast', 't1_normal', 't2_flair','t2_weighted'")

def index_to_name_DEPRECATED(index: int) -> str:
    '''
    example of order
    BraTS-GLI-01368-000-seg.nii.gz
    BraTS-GLI-01368-000-t1c.nii.gz
    BraTS-GLI-01368-000-t1n.nii.gz
    BraTS-GLI-01368-000-t2f.nii.gz
    BraTS-GLI-01368-000-t2w.nii.gz
    '''
    match index:
        case 0:
            return 'segmentation'
        case 1:
            return 't1_contrast'
        case 2:
            return 't1_normal'
        case 3:
            return 't2_flair'
        case 4:
            return 't2_weighted'
        case _:
            raise ValueError("Index must be between 0 and 4")

@deprecated("Using 0 for the segmentation is bad and confusing")
def intex_to_nnunet_name_DEPRECATED(index:int)->str:
    match index:
        case 1:
            return '0001'
        case 2:
            return '0002'
        case 3:
            return '0003'
        case 4:
            return '0000'
        case _:
            raise ValueError("Index must be between 1 and 4")

def index_to_nnunet_name(index:int)->str:
    match index:
        case 0:
            return '0001'
        case 1:
            return '0002'
        case 2:
            return '0003'
        case 3:
            return '0000'
        case _:
            raise ValueError("Index must be between 0 and 3")
        
def index_to_name(index:int)->str:
        match index:
            case 0:
                return 't1_contrast'
            case 1:
                return 't1_normal'
            case 2:
                return 't2_flair'
            case 3:
                return 't2_weighted'
            case _:
                raise ValueError("Index must be between 0 and 4")
        
def conv_path_nnunet_to_brats(nnunet_path:str|Path)->Path:
    '''
    This function converts whatever nnunet-style path to its relative brats-style version.
    Ex:
    /path/to/images/BraTS-GLI-00712-000_0003.nii.gz     --> BraTS-GLI-00712-000/BraTS-GLI-00712-000-t2f.nii.gz

    Or directly, with relative paths:
    to/images/BraTS-GLI-00712-000_0002.nii.gz           --> BraTS-GLI-00712-000/BraTS-GLI-00712-000-t1n.nii.gz
    BraTS-GLI-00712-000_0000.nii.gz                     --> BraTS-GLI-00712-000/BraTS-GLI-00712-000-t2w.nii.gz
    The return type will always be a Path
    '''
    path = str(Path(nnunet_path).name) # Making it relative
    path_splitted = path.split('_')
    id = path_splitted[0]
    end = path_splitted[1]

    end = end.split('.')[0]
    
    conversion_dict = {
        '0001':'t1c',
        '0002':'t1n',
        '0003':'t2f',
        '0000':'t2w',
    }
    new_end = conversion_dict[end]

    brats_path = Path(id)/ (id + '-' + new_end + '.nii.gz')
    return brats_path
    
def conv_path_nnunet_to_nnunet_segm(nnunet_path:str|Path)->Path:
    '''
    This function converts whatever nnunet-style path to its relative nnunet-style segmentation version.
    Ex:
    /path/to/images/BraTS-GLI-00712-000_0003.nii.gz     --> BraTS-GLI-00712-000.nii.gz

    Or directly, with relative paths:
    to/images/BraTS-GLI-00712-000_0002.nii.gz           --> BraTS-GLI-00712-000.nii.gz
    BraTS-GLI-00712-000_0000.nii.gz                     --> BraTS-GLI-00712-000.nii.gz
    The return type will always be a Path
    '''
    path = str(Path(nnunet_path).name) # Making it relative
    path_splitted = path.split('_')
    id = path_splitted[0]
    
    return Path(id + '.nii.gz')