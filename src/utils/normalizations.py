import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from src.utils.indexing import index_to_name_DEPRECATED
from src.data.dataset import BrainDataset3D,brain_collate_stable_v1
from pathlib import Path
from torch.utils.data import DataLoader
from typing import List, Tuple
from tqdm import tqdm
from math import sqrt
from multiprocessing import Pool

def plot_stats_complete(csv_path:str|Path = '/home/user/output/csvs/dataset_stats_complete_zero_clipped.csv'):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(4, 4, figsize=(12, 6))
    axes = axes.flatten()
    df_without_ids = df.iloc[:,2:]
    for i, col in enumerate(df_without_ids.columns):
        axes[i].boxplot(df_without_ids[col])
        axes[i].set_title(f"{col}")
        axes[i].set_ylabel('vals')

    plt.title('Stats')
    plt.tight_layout()
    plt.show()
    
def create_stats_complete(collate,out_path):
    dataset = BrainDataset3D(dataset_dirs=['/work_path/brats3d/train_gli',
                                         '/work_path/brats3d/train_met',
                                         '/work_path/brats3d/train_met_add'])
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size = 1,num_workers=16,collate_fn=collate)
    cols = ['id'] + [elem for mod in ('t1_contrast','t1_normal','t2_flair','t2_weighted') for elem in (mod + '_min',mod +'_max',mod +'_mean',mod + '_std')]
    def_vals = [''] + [0.0 for _ in range(len(cols) - 1)]
    all_rows = []
    for brain_unsqueezed, id in tqdm(dataloader):
        if brain_unsqueezed is None:
            print(f'this-->{id} is broken')
            quit()
        brain = brain_unsqueezed.squeeze(0)
        row = pd.DataFrame({
            key : val for key,val in zip(cols,def_vals)},index=[0])
        row.iloc[0,0] = id[0]
        for i in range(1,5):
            try:
                modality = brain[i]
                modality = modality[modality != 0]
                row.iloc[0,(i-1)*4 + 1] = modality.min().item()
                row.iloc[0,(i-1)*4 + 2] = modality.max().item()
                row.iloc[0,(i-1)*4 + 3] = modality.mean().item()
                row.iloc[0,(i-1)*4 + 4]= modality.std().item()
            except:
                print(i)
        all_rows.append(row)

    df = pd.concat(all_rows,ignore_index=True)
    df.to_csv(out_path,index=False)
def show_brain_already_sliced(slice_tensor:torch.Tensor,desc:str):
    # Plot the slice
    slice_data = slice_tensor.detach().to(torch.device('cpu')).numpy()
    plt.imshow(slice_data.T, cmap='gray_r', origin='lower')
    desc = f'{desc}'
    plt.title(f"{desc}")
    plt.axis('off')
    plt.show()
    
def show_brain_from_tensor(brain_tensor:torch.Tensor, desc:str, index = None, brain_slices = None):
    if slice is None:
        brain_slices = (0, 1, 120, slice(None), slice(None))
    assert len(brain_slices.shape) == 5, 'Please provice a 5 dimensional slice'
    num_ints = 0
    for elem in brain_slices:
        if isinstance(elem,int):
            num_ints += 1
    assert num_ints == 3, 'Please provide slices with exacty 3 integers out of 5 elements'
    slice_data = brain_tensor.to(torch.device('cpu')).numpy()[brain_slices]

    # Plot the slice
    plt.imshow(slice_data.T, cmap='gray_r', origin='lower')
    desc = f'{desc},{index_to_name_DEPRECATED(index)}' if index is not None else f'{desc}'
    plt.title(f"{desc}")
    plt.axis('off')
    plt.show()

def show_brain_from_nii(path: str, index=None):
    # Load the NIfTI image
    img = nib.load(path)
    data = img.get_fdata()

    # Select a central slice (adjust axis as needed)
    slice_data = data[120, :, :]

    # Plot the slice
    plt.imshow(slice_data.T, cmap='gray_r', origin='lower')
    desc = f'{path},{index_to_name_DEPRECATED(index)}' if index is not None else f'{path}'
    plt.title(f"{desc}")
    plt.axis('off')
    plt.show()
    
def create_histograms():
    '''BraTS-GLI-01368-000-t1c.nii.gz
    BraTS-GLI-01368-000-t1n.nii.gz
    BraTS-GLI-01368-000-t2f.nii.gz
    BraTS-GLI-01368-000-t2w.nii.gz'''
    t1c_max = 5_000_000
    t1n_max = 5_000_000
    t2f_max = 5_000_000
    t2w_max = 5_000_000
    maxes = [t1c_max, t1n_max, t2f_max, t2w_max]

    # Initialize histograms
    histograms = [
        torch.zeros(t1c_max +1),
        torch.zeros(t1n_max +1),
        torch.zeros(t2f_max +1),
        torch.zeros(t2w_max +1)
    ]

    dataset = BrainDataset3D(dataset_dirs=['/work_path/brats3d/train_gli',
                                         '/work_path/brats3d/train_met',
                                         '/work_path/brats3d/train_met_add',
                                         '/work_path/brats3d/train_men'])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=brain_collate_stable_v1,
        shuffle=True,
        num_workers=8
    )

    for brains_un, paths in tqdm(dataloader):
        brains = brains_un.squeeze(0)
        for i in range(4):  # Corrected indexing: 0-3 instead of 1-4
            histograms[i] += torch.histc(brains[i+1],min = 0,max = maxes[i],bins=maxes[i]+1)

    print('end of iter')
    return histograms

def load_all_histograms()->List[torch.Tensor]:
    loaded = []
    for i in range(4):
        loaded.append(torch.load(f'BraSyn25/tensors/histogram_{index_to_name_DEPRECATED(i+1)}.pt'))
    return loaded
def load_one_histogram(path:Path)->List[torch.Tensor]:
    loaded = torch.load(path)
    return loaded.unsqueeze(0)

def _hard_coded_clamp(brains:torch.Tensor):
    upper_bounds = [4623, 3503, 1797, 2532]
    for i in range(1,5):
        brains[i] = brains[i].clamp(min = 0, max=upper_bounds[i - 1])
    return brains

def _scatter_perversion(tensor:torch.Tensor,title):
    # Filter out zero values (create a mask for non-zero elements)
    non_zero_mask = tensor != 0

    x = torch.nonzero(non_zero_mask).squeeze()

    y = tensor[non_zero_mask]

    # Create scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(x.numpy(), y.numpy(), alpha=0.7, color='b', s=50)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("n of occurrences")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def _hard_coded_slices_visual(index:int):
    assert index > 0 and index < 5, 'index out of range'
    upper_bounds = [4623, 3503, 1797, 2532]
    return slice(1,upper_bounds[index-1])

def show_all_scatter_perversions():
    tensor_dir = Path('/work_path/tesi_proj/output/tensors')
    for i,tensor_file in enumerate(tensor_dir.iterdir()):
        _scatter_perversion(torch.load(str(tensor_file))[_hard_coded_slices_visual(i+1)],index_to_name_DEPRECATED(i+1))

def scatter_perversion_but_together():
    tensor_dir = Path('/work_path/tesi_proj/output/tensors')
    colors = ['r','g','b','y']
    plt.figure(figsize=(12, 10))
    plt.xlabel("value")
    plt.ylabel("n of occurrences")
    plt.grid(True)
    plt.tight_layout()

    for i,tensor_file in enumerate(tensor_dir.iterdir()):
        color = colors[i]
        tensor = torch.load(str(tensor_file))[_hard_coded_slices_visual(i+1)]
        non_zero_mask = tensor != 0
        x = torch.nonzero(non_zero_mask).squeeze()
        y = tensor[non_zero_mask]
        plt.scatter(x.numpy(), y.numpy(), alpha=0.7, color=color, s=50,label = f'{index_to_name_DEPRECATED(i+1)}')
    plt.legend(title="modalities", loc="best")
    plt.show()
    
def get_percentile_max_val(histograms:torch.Tensor,cut:float)->int | int:
    max_val = 0
    assert cut >= 0 and cut < 100, 'invalid value for cut'
    total_sum = torch.sum(histograms[1:])
    actual_sum = 0
    target_sum = total_sum * (100 - cut)/100
    for value, bin in enumerate(histograms[1:].flip(0)):
        actual_sum += bin.item()
        if actual_sum >= target_sum:
            max_val = histograms[1:].size(0) - 1 - value
            break

    return max_val


def get_mean_std_from_tensor_histogram(histogram: torch.Tensor,max_val:float)->Tuple[float,float]:
    bins = torch.arange(histogram.shape[0])
    bins = torch.clamp(bins, max=max_val)
    
    # Skipping zero (background)
    weights = histogram[1:]  
    bins = bins[1:]

    sum = torch.sum(weights)
    mean = torch.sum(bins * weights) / sum
    variance = torch.sum(weights * (bins - mean) ** 2) / sum
    std = torch.sqrt(variance)
    
    return float(mean), float(std)

def get_mean_std_from_tensor_path(root_dir:str|Path = 'BraSyn25/tensors',cut:float = 98.0)->Tuple[torch.Tensor,torch.Tensor]:
    root_dir = Path(root_dir)
    all_means = torch.zeros((4))
    all_stds = torch.zeros((4))
    for i in tqdm(range(1,5)):
        tensor_path = root_dir / Path('histogram_' + index_to_name_DEPRECATED(i) + '.pt')
        tensor = torch.load(tensor_path)
        max_val = get_percentile_max_val(tensor,cut)
        all_means[i-1], all_stds[i-1] = get_mean_std_from_tensor_histogram(tensor,max_val)
    return all_means, all_stds


def process_tensor_index(args):
    i, root_dir, cut,verbose = args
    tensor_path = root_dir / Path(f'histogram_{index_to_name_DEPRECATED(i)}.pt')
    tensor = torch.load(tensor_path)
    max_val = get_percentile_max_val(tensor, cut)
    mean, std = get_mean_std_from_tensor_histogram(tensor, max_val)
    if verbose:
        print(f'Finished cut {i}')
    return mean, std, max_val


def get_stats_from_path(root_dir: str | Path = 'BraSyn25/tensors',
                                           cut: float = 98.0,verbose:bool = False):

    root_dir = Path(root_dir)

    # Prepare arguments for each process
    args = [(i, root_dir, cut,verbose) for i in range(1, 5)]
    if verbose:
        print('Starting the cut')
    # Use Pool instead of ProcessPoolExecutor for simplicity
    with Pool(processes=4) as pool:
        results = pool.map(process_tensor_index, args)

    # Convert results to tensors
    all_means = torch.tensor([mean for mean, _,_ in results])
    all_stds = torch.tensor([std for _, std,_ in results])
    all_max_vals = torch.tensor([max_val for _,_,max_val in results])
    return all_means, all_stds, all_max_vals