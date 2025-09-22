import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import List,Sequence
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import random


def get_tensor_header_affine_from_nii(path: Path):
    nii_img = nib.load(path)
    nii_data = np.ascontiguousarray(nii_img.get_fdata(dtype=np.float32))
    nii_tensor = torch.from_numpy(nii_data).to(torch.float32)
    nii_header = nii_img.header
    nii_affine = nii_img.affine
    return nii_tensor, nii_header, nii_affine


def save_tensor_to_nii(tensor, filename, affine=None, header = None):
    # Check shape
    if tensor.shape != (1, 240, 240, 155):
        raise ValueError(f"Expected shape (1,240,240,155), got {tensor.shape}")

    # Remove batch dimension and convert to numpy
    np_array = tensor.squeeze(0).cpu().numpy()
    if affine is None:
        affine = np.eye(4)

    if header is None:
        nii_img = nib.Nifti1Image(np_array, affine)
    else:
        nii_img = nib.Nifti1Image(np_array, affine, header)


    # Save to file
    nib.save(nii_img, filename)
    #print(f"Saved to {filename}")


class BrainDataset3D(Dataset):
    def __init__(self, dataset_dirs: str | Path | Sequence[Path] | list[Path] | list[str],
                 split: str | Path | Sequence[Path] | list[Path] | list[str] | None = None,
                 verbose=False,

                 ):
        # Parameter checking and stuff
        if isinstance(dataset_dirs, Path):
            _dataset_dirs = [dataset_dirs]
        elif isinstance(dataset_dirs, str):
            _dataset_dirs = [Path(dataset_dirs)]
        elif isinstance(dataset_dirs, Sequence):
            _dataset_dirs = [Path(directory) for directory in dataset_dirs]
        else:
            raise TypeError("paths must be a string, Path or a Sequence")

        for path in _dataset_dirs:
            assert path.is_absolute(), f'{path} is not an absolute path'

        self._values: list[Path] = []
        if split is not None:
            if isinstance(split, str) or isinstance(split, Path):
                splits = [split]
            else:
                splits = split
            split_list = []
            for s in splits:
                split_df = pd.read_csv(s)
                split_list += [element[-9:] for element in split_df['case'].to_list()]
            split_set = frozenset(split_list)
            for dataset_dir in _dataset_dirs:
                if verbose:
                    iterable = tqdm(list(dataset_dir.iterdir()), total=len(list(dataset_dir.iterdir())))
                else:
                    iterable = dataset_dir.iterdir()
                self._values += [file for file in iterable if file.is_dir() and file.name[-9:] in split_set]
        else:
            for dataset_dir in _dataset_dirs:
                self._values += [file for file in dataset_dir.iterdir() if file.is_dir()]

    @classmethod
    def init_from_values(cls, values: List):
        obj = cls.__new__(cls)
        obj._values = values
        return obj

    def __add__(self, other):  # type: ignore[reportIncompatibleMethodOverride]
        total_values = self._values + other._values
        return type(self).init_from_values(total_values)

    def __getitem__(self, index):
        base_dir = self._values[index]
        paths = [element for element in base_dir.iterdir() if str(element).endswith('.nii.gz')]
        return paths

    def __len__(self):
        return len(self._values)


def ULTIMATE_brain_coll_3d(batch):
    '''
    This collate function will be useful for evaluation
    Usage:
    '''
    all_ids = []
    all_brains_tensor_list = []
    all_segmentations_tensor_list = []

    all_brain_affines = []
    all_segmentation_affines = []

    all_brain_headers = []
    all_segmentation_headers = []

    all_missing_modals_list = []

    # Normalization stats
    means = torch.tensor([1066.3375, 781.2247, 510.9852, 673.4393])
    stds = torch.tensor([1301.7006, 944.3418, 769.4159, 804.3864])
    max_vals = torch.tensor([8664, 7315, 8842, 8233])
    mins = (torch.zeros_like(means) - means) / stds
    name_to_label_dict = {
        'seg.nii.gz': 0,
        't1c.nii.gz': 1,
        't1n.nii.gz': 2,
        't2f.nii.gz': 3,
        't2w.nii.gz': 4

    }
    for brains in batch:  # For each subject in the batch
        samebrain_tensor_list: List[torch.Tensor] = []  # List of all the modalities for the same brain
        samebrain_affines = []  # List of all the affine matrices
        samebrain_headers = []  # List of all the nifti headers
        missing_modal = -1

        seg_header = None
        seg_affine = None

        id = str(brains[0].parent)
        all_ids.append(
            id)  # Getting the path of the directory of the specific patient. ex: /work/tesi_ocarpentiero/train_gli/BraTS-GLI-00000-000

        # This sections iterates over the dictionary, finding and processing the modalities in the right order
        for key, label in name_to_label_dict.items():
            brain = [elem for elem in brains if key in str(elem)]
            assert len(brain) <= 1, f'Why did i find two files with the same modality??????????????-->{id} ???????????'
            if len(brain) == 0:  # Missing modality/segmentation
                brain_tensor = torch.full(size=(240, 240, 155), fill_value=-1)
                header, affine = None, None
                if label != 0:
                    missing_modal = label - 1
            else:
                brain_tensor, header, affine = get_tensor_header_affine_from_nii(
                    brain[0])  # conversion to tensor# conversion to tensor
                # Normalization (skipped for the segmentation)
                if label != 0:
                    brain_tensor = brain_tensor.clamp(min=0, max=max_vals[label - 1].item())
                    brain_tensor = (brain_tensor - means[label - 1]) / stds[label - 1]
                    brain_tensor[brain_tensor <= mins[label - 1]] = -1
            if label == 0:  # Segmentation
                all_segmentations_tensor_list.append(brain_tensor.unsqueeze(0).clone())
                seg_header = header
                seg_affine = affine
            else:
                samebrain_tensor_list.append(brain_tensor.clone())
                samebrain_affines.append(affine)
                samebrain_headers.append(header)

        samebrain_tensor = torch.stack(samebrain_tensor_list, dim=0)  # stacking the modalities on a single tensor
        all_brains_tensor_list.append(samebrain_tensor)

        if missing_modal == -1:
            missing_modal = random.randint(0, 3)
        all_missing_modals_list.append(torch.tensor(missing_modal))

        available_affine = None
        available_header = None
        for aff in samebrain_affines + [seg_affine]:
            if aff is not None:
                available_affine = aff
        for head in samebrain_headers + [seg_header]:
            if head is not None:
                available_header = head

        assert available_header is not None, 'No non-None header found????'
        assert available_affine is not None, 'No non-None affine found????'

        if seg_affine is None:
            seg_affine = available_affine
        if seg_header is None:
            seg_header = available_header

        for i, aff in enumerate(samebrain_affines):
            if aff is None:
                samebrain_affines[i] = available_affine
        for i, head in enumerate(samebrain_headers):
            if head is None:
                samebrain_headers[i] = available_header

        all_brain_affines.append(samebrain_affines)
        all_brain_headers.append(samebrain_headers)

        all_segmentation_affines.append(seg_affine)
        all_segmentation_headers.append(seg_header)

    all_brains_tensor = torch.stack(all_brains_tensor_list, dim=0).contiguous()
    all_segmentations_tensor = torch.stack(all_segmentations_tensor_list, dim=0).contiguous()
    all_missing_modals = torch.stack(all_missing_modals_list)
    return {
        'images': all_brains_tensor,
        'image_headers': all_brain_headers,
        'image_affines': all_brain_affines,

        'segmentations': all_segmentations_tensor,
        'segmentation_headers': all_segmentation_headers,
        'segmentation_affines': all_segmentation_affines,

        'missing_modalities': all_missing_modals,
        'ids': all_ids
    }