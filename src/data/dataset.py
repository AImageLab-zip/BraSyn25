# External imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pathlib import Path
import os
from typing import List, Sequence, Tuple, Optional, overload
import gzip
import random
from PIL import Image
import pandas as pd
from tqdm import tqdm

# My imports
from src.utils.preprocessing import path_to_nii_to_tensor, hard_coded_clamp, get_tensor_header_affine_from_nii

# Typing
PathLike = str | Path
MultiPathLike = PathLike | Sequence[PathLike]

class BrainDataset3D(Dataset):
    def __init__(
            self,
            dataset_dirs: MultiPathLike,
            split: MultiPathLike | None = None,
            verbose: bool = False,
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


class BrainDataset2D(Dataset):
    def __init__(self, dataset_dirs: str | Path | Sequence[Path] | list[Path] = [
        Path('/work_path/brats2d_axi/train_gli')],
                 split: Optional[str] | Optional[List[str]] = None,
                 verbose: bool = False
                 ):
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
            if isinstance(split, str):
                split = [split]
            split_list = []
            for s in split:
                split_df = pd.read_csv(s)
                split_list += [element[-9:] for element in split_df['case'].to_list()]
            split_set = frozenset(split_list)
            for dataset_dir in _dataset_dirs:
                if verbose:
                    iterable = tqdm(list(dataset_dir.iterdir()), total=len(list(dataset_dir.iterdir())))
                else:
                    iterable = dataset_dir.iterdir()
                self._values += [file for file in iterable if not file.is_dir() and file.name[10:19] in split_set]
        else:
            for dataset_dir in _dataset_dirs:
                self._values += [file for file in dataset_dir.iterdir() if not file.is_dir()]
        self._values.sort()

    @classmethod
    def init_from_values(cls, values: List):
        obj = cls.__new__(cls)
        obj._values = values
        return obj

    def __add__(self, other):  # type: ignore[reportIncompatibleMethodOverride]
        total_values = self._values + other._values
        return type(self).init_from_values(total_values)

    def __getitem__(self, index):
        item = self._values[index]
        return item

    def __len__(self):
        return len(self._values)



class BrainCollDyn3D:
    def __init__(self, cut: float = 99.5, verbose: bool = True,
                 root_dir: Path | str = 'BraSyn25/tensors'):
        from src.utils.normalizations import get_stats_from_path
        import wandb
        root_dir = Path(root_dir)
        if cut == 99.5:
            self.means = torch.tensor([1066.3375, 781.2247, 510.9852, 673.4393])
            self.stds = torch.tensor([1301.7006, 944.3418, 769.4159, 804.3864])
            self.max_vals = torch.tensor([8664, 7315, 8842, 8233])
        else:
            self.means, self.stds, self.max_vals = get_stats_from_path(root_dir=root_dir, cut=cut, verbose=verbose)
        self.mins = (torch.zeros_like(self.means) - self.means) / self.stds
        if verbose:
            print(f'means-->{self.means}')
            print(f'stds-->{self.stds}')
            print(f'max_vals-->{self.max_vals}')

        '''if wandb.run is not None:
            wandb.config.update({"means": self.means,
                                 "stds": self.stds,
                                 "max_vals": self.max_vals})
        else:
            print('No wandb run found')'''

    def __call__(self, batch):
        all_ids = []
        all_brains_tensor_list = []
        all_labels = []
        name_to_label_dict = {
            'seg.nii.gz': 0,
            't1c.nii.gz': 1,
            't1n.nii.gz': 2,
            't2f.nii.gz': 3,
            't2w.nii.gz': 4

        }
        for brains in batch:
            samebrain_tensor_list: List[torch.Tensor] = []  # list of all the modalities for the same brain
            samebrain_labels: List[torch.Tensor] = []  # list of
            all_ids.append(brains[
                               0].parent)  # getting the path of the directory of the specific patient. ex: /work_path/train_gli/BraTS-GLI-00000-000

            # this sections iterates over the dictionary, finding and processing the modalities in the right order
            for key, label in name_to_label_dict.items():
                brain = [elem for elem in brains if key in str(elem)]
                assert len(
                    brain) <= 1, f'Dataloader found {len(brain)} brains in {all_ids[-1]}'  # always one or zero brain per modality
                if len(brain) == 0:  # missing modality
                    continue
                else:
                    samebrain_labels.append(torch.tensor(label))
                    brain = brain[0]  # list to element
                    brain_tensor = path_to_nii_to_tensor(brain)  # conversion to tensor
                    if label != 0:  # do not normalize if it is the segmentation mask
                        brain_tensor = brain_tensor.clamp(min=0, max=self.max_vals[label - 1].item())
                        brain_tensor = (brain_tensor - self.means[label - 1]) / self.stds[label - 1]
                        brain_tensor[brain_tensor == self.mins[label - 1]] = -1
                    samebrain_tensor_list.append(brain_tensor)
            samebrain_tensor = torch.stack(samebrain_tensor_list)  # stacking the modalities on a single tensor
            samebrain_labels_tensor = torch.stack(samebrain_labels)
            all_brains_tensor_list.append(samebrain_tensor)
            all_labels.append(samebrain_labels_tensor)
        all_brains_tensor = torch.stack(all_brains_tensor_list)
        all_labels_tensor = torch.stack(all_labels)
        return all_brains_tensor, all_labels_tensor, all_ids


class BrainCollDyn2DMask:
    def __init__(self, cut: float = 99.5, verbose: bool = True,
                 root_dir: Path | str = 'BraSyn25/tensors'):
        from src.utils.normalizations import get_stats_from_path
        import wandb
        import pandas as pd
        import torch.nn.functional as F

        root_dir = Path(root_dir)
        if cut == 99.5:
            self.means = torch.tensor([1066.3375, 781.2247, 510.9852, 673.4393])
            self.stds = torch.tensor([1301.7006, 944.3418, 769.4159, 804.3864])
            self.max_vals = torch.tensor([8664, 7315, 8842, 8233])
        else:
            self.means, self.stds, self.max_vals = get_stats_from_path(root_dir=root_dir, cut=cut, verbose=verbose)
        self.mins = (torch.zeros_like(self.means) - self.means) / self.stds
        if verbose:
            print(f'means-->{self.means}')
            print(f'stds-->{self.stds}')
            print(f'max_vals-->{self.max_vals}')

        if wandb.run is not None:
            wandb.config.update({"means": self.means,
                                 "stds": self.stds,
                                 "max_vals": self.max_vals})
        else:
            print('No wandb run found')

    def __call__(self, batch):
        image_list = []
        image_masked_list = []
        target_list = []
        missing_modality_list = []

        for path in batch:
            brain_slice = path_to_nii_to_tensor(path)
            if brain_slice.shape[0] == 5:
                brain_slice = brain_slice[1:]

            for mod in range(brain_slice.shape[0]):  # range(4)
                brain_slice[mod, :, :] = torch.clamp(brain_slice[mod, :, :], min=0, max=self.max_vals[mod].item())

            brain_slice = (brain_slice - self.means.reshape(-1, 1, 1)) / self.stds.reshape(-1, 1, 1)

            for mod in range(brain_slice.shape[0]):  # range(4)
                brain_slice[mod, :, :][brain_slice[mod, :, :] <= self.mins[mod]] = -1

            image_list.append(brain_slice.clone())

            missing_modality = random.randint(0, 3)
            missing_modality_list.append(torch.tensor(missing_modality))

            masked_slice = brain_slice.clone()
            masked_slice[missing_modality] = -1
            image_masked_list.append(masked_slice)

            missing_modality_slice = brain_slice.clone()[missing_modality, :, :].unsqueeze(0)
            target_list.append(missing_modality_slice)
        image = torch.stack(image_list, dim=0)
        image_masked = torch.stack(image_masked_list)
        target = torch.stack(target_list)
        missing_modalities = torch.stack(missing_modality_list)
        return {
            'image': image,  # (B,4,240,240)
            'image_masked': image_masked,  # (B,4,240,240)
            'target': target,  # (B,1,240,240)
            'modality': missing_modalities  # (B)
        }


def brain_coll_dyn_2d_mask_seg(batch):
    means = torch.tensor([1066.3375, 781.2247, 510.9852, 673.4393])
    stds = torch.tensor([1301.7006, 944.3418, 769.4159, 804.3864])
    max_vals = torch.tensor([8664, 7315, 8842, 8233])
    mins = (torch.zeros_like(means) - means) / stds

    image_list = []
    image_masked_list = []
    target_list = []
    missing_modality_list = []
    views_list = []
    segmentations_list = []
    for path in batch:
        brain_slice = path_to_nii_to_tensor(path)

        if 'axi' in str(path):
            view = 0
        elif 'sag' in str(path):
            view = 1
        elif 'cor' in str(path):
            view = 2
        else:
            raise RuntimeError("The path doesn't contain 'axi'/'cor'/'sag'...so strange!")
        views_list.append(torch.tensor(view))
        if brain_slice.shape[0] == 5:
            segmentation_slice = brain_slice[0:1].clone()
            segmentations_list.append(segmentation_slice)

            brain_slice = brain_slice[1:]
        else:
            raise RuntimeError('You used BrainCollDyn2DMaskSeg for validation-->bad!')

        for mod in range(brain_slice.shape[0]):  # range(4)
            brain_slice[mod, :, :] = torch.clamp(brain_slice[mod, :, :], min=0, max=max_vals[mod].item())

        brain_slice = (brain_slice - means.reshape(-1, 1, 1)) / stds.reshape(-1, 1, 1)

        for mod in range(brain_slice.shape[0]):  # range(4)
            brain_slice[mod, :, :][brain_slice[mod, :, :] <= mins[mod]] = -1

        image_list.append(brain_slice.clone())

        missing_modality = random.randint(0, 3)
        missing_modality_list.append(torch.tensor(missing_modality))

        masked_slice = brain_slice.clone()
        masked_slice[missing_modality] = -1
        image_masked_list.append(masked_slice)

        missing_modality_slice = brain_slice.clone()[missing_modality, :, :].unsqueeze(0)
        target_list.append(missing_modality_slice)
    image = torch.stack(image_list, dim=0)
    image_masked = torch.stack(image_masked_list)
    target = torch.stack(target_list)
    segmentations = torch.stack(segmentations_list, dim=0)
    missing_modalities = torch.stack(missing_modality_list)
    views = torch.stack(views_list)
    return {
        'image': image,  # (B,4,240,240)
        'image_masked': image_masked,  # (B,4,240,240)
        'target': target,  # (B,1,240,240)
        'modality': missing_modalities,  # (B)
        'segmentations': segmentations,  # (B,1,240,240)
        'paths': batch,
        'views': views
    }


def brain_coll_seg_gen(batch):
    means = torch.tensor([1066.3375, 781.2247, 510.9852, 673.4393])
    stds = torch.tensor([1301.7006, 944.3418, 769.4159, 804.3864])
    max_vals = torch.tensor([8664, 7315, 8842, 8233])
    mins = (torch.zeros_like(means) - means) / stds

    image_list = []
    segmentations_list = []
    for path in batch:
        brain_slice = path_to_nii_to_tensor(path)

        if brain_slice.shape[0] == 5:
            segmentation_slice = brain_slice[0:1].clone()
            segmentations_list.append(segmentation_slice)

            brain_slice = brain_slice[1:]
        else:
            raise RuntimeError('Where are the segmentations???????????????????????????????????????')

        for mod in range(brain_slice.shape[0]):  # range(4)
            brain_slice[mod, :, :] = torch.clamp(brain_slice[mod, :, :], min=0, max=max_vals[mod].item())

        brain_slice = (brain_slice - means.reshape(-1, 1, 1)) / stds.reshape(-1, 1, 1)

        for mod in range(brain_slice.shape[0]):  # range(4)
            brain_slice[mod, :, :][brain_slice[mod, :, :] <= mins[mod]] = -1

        image_list.append(brain_slice.clone())

    image = torch.stack(image_list, dim=0)
    segmentations = torch.stack(segmentations_list, dim=0)
    return {
        'image': image,  # (B,4,240,240)
        'segmentations': segmentations  # (B,1,240,240)
    }


def brain_coll_seg_gen_fusion(batch):
    means = torch.tensor([1066.3375, 781.2247, 510.9852, 673.4393])
    stds = torch.tensor([1301.7006, 944.3418, 769.4159, 804.3864])
    max_vals = torch.tensor([8664, 7315, 8842, 8233])
    mins = (torch.zeros_like(means) - means) / stds

    image_list = []
    segmentations_list = []
    for path in batch:
        brain_slice = path_to_nii_to_tensor(path)
        if brain_slice.shape[0] == 5:
            segmentation_slice = brain_slice[0:1].clone()
            segmentations_list.append(segmentation_slice)

            brain_slice = brain_slice[1:]
        else:
            raise RuntimeError('Where are the segmentations???????????????????????????????????????')
        infuse_tensor_1 = torch.zeros((1, brain_slice.shape[1], brain_slice.shape[2]), dtype=torch.float32)
        infuse_tensor_2 = torch.ones_like(infuse_tensor_1, dtype=torch.float32)
        if 'met' in str(path):
            infuse_tensor = torch.cat((infuse_tensor_1, infuse_tensor_2), dim=0)
        else:
            infuse_tensor = torch.cat((infuse_tensor_2, infuse_tensor_1), dim=0)

        for mod in range(brain_slice.shape[0]):  # range(4)
            brain_slice[mod, :, :] = torch.clamp(brain_slice[mod, :, :], min=0, max=max_vals[mod].item())

        brain_slice = (brain_slice - means.reshape(-1, 1, 1)) / stds.reshape(-1, 1, 1)

        for mod in range(brain_slice.shape[0]):  # range(4)
            brain_slice[mod, :, :][brain_slice[mod, :, :] <= mins[mod]] = -1
        brain_slice = torch.cat((brain_slice, infuse_tensor), dim=0)
        image_list.append(brain_slice.clone())

    image = torch.stack(image_list, dim=0).contiguous()
    segmentations = torch.stack(segmentations_list, dim=0).contiguous()
    return {
        'image': image,  # (B,6,240,240)
        'segmentations': segmentations  # (B,1,240,240)
    }


def brain_coll_seg_total_fusion(batch):
    means = torch.tensor([1066.3375, 781.2247, 510.9852, 673.4393])
    stds = torch.tensor([1301.7006, 944.3418, 769.4159, 804.3864])
    max_vals = torch.tensor([8664, 7315, 8842, 8233])
    mins = (torch.zeros_like(means) - means) / stds

    image_list = []
    segmentations_list = []
    for path in batch:
        brain_slice = path_to_nii_to_tensor(path)
        if brain_slice.shape[0] == 5:
            segmentation_slice = brain_slice[0:1]
            segmentations_list.append(segmentation_slice)

            brain_slice = brain_slice[1:]
        else:
            raise RuntimeError('Where are the segmentations???????????????????????????????????????')

        infuse_tum = torch.zeros((2, brain_slice.shape[1], brain_slice.shape[2]), dtype=brain_slice.dtype)
        if 'met' in str(path):
            infuse_tum[0] = 1
        elif 'gli' in str(path):
            infuse_tum[1] = 1
        else:
            raise RuntimeError(f'Found an unknown tumor type in path: {str(path)}')

        infuse_view = torch.zeros((3, brain_slice.shape[1], brain_slice.shape[2]), dtype=brain_slice.dtype)
        if 'axi' in str(path):
            infuse_view[0] = 1.0
        elif 'sag' in str(path):
            infuse_view[1] = 1.0
        elif 'cor' in str(path):
            infuse_view[2] = 1.0
        else:
            raise RuntimeError(f'Found an unknown view in path: {str(path)}')

        for mod in range(brain_slice.shape[0]):  # range(4)
            brain_slice[mod, :, :] = torch.clamp(brain_slice[mod, :, :], min=0, max=max_vals[mod].item())

        brain_slice = (brain_slice - means.reshape(-1, 1, 1)) / stds.reshape(-1, 1, 1)

        for mod in range(brain_slice.shape[0]):  # range(4)
            brain_slice[mod, :, :][brain_slice[mod, :, :] <= mins[mod]] = -1
        brain_slice = torch.cat((brain_slice, infuse_tum, infuse_view), dim=0)
        image_list.append(brain_slice)

    image = torch.stack(image_list, dim=0).contiguous()
    segmentations = torch.stack(segmentations_list, dim=0).contiguous()
    return {
        'image': image,  # (B,9,240,240)
        'segmentations': segmentations  # (B,1,240,240)
    }


class BrainCollDyn2DMaskSeg:
    def __init__(self, cut: float = 99.5, verbose: bool = True,
                 root_dir: Path | str = 'BraSyn25/tensors'):
        from src.utils.normalizations import get_stats_from_path
        import wandb
        import pandas as pd
        import torch.nn.functional as F

        root_dir = Path(root_dir)
        if cut == 99.5:
            self.means = torch.tensor([1066.3375, 781.2247, 510.9852, 673.4393])
            self.stds = torch.tensor([1301.7006, 944.3418, 769.4159, 804.3864])
            self.max_vals = torch.tensor([8664, 7315, 8842, 8233])
        else:
            self.means, self.stds, self.max_vals = get_stats_from_path(root_dir=root_dir, cut=cut, verbose=verbose)
        self.mins = (torch.zeros_like(self.means) - self.means) / self.stds
        if verbose:
            print(f'means-->{self.means}')
            print(f'stds-->{self.stds}')
            print(f'max_vals-->{self.max_vals}')

        '''if wandb.run is not None:
            wandb.config.update({"means": self.means,
                                 "stds": self.stds,
                                 "max_vals": self.max_vals})
        else:
            print('No wandb run found')'''

    def __call__(self, batch):
        image_list = []
        image_masked_list = []
        target_list = []
        missing_modality_list = []
        segmentations_list = []
        for path in batch:
            brain_slice = path_to_nii_to_tensor(path)

            if brain_slice.shape[0] == 5:
                segmentation_slice = brain_slice[0:1].clone()
                segmentations_list.append(segmentation_slice)

                brain_slice = brain_slice[1:]
            else:
                raise RuntimeError('You used BrainCollDyn2DMaskSeg for validation-->bad!')

            for mod in range(brain_slice.shape[0]):  # range(4)
                brain_slice[mod, :, :] = torch.clamp(brain_slice[mod, :, :], min=0, max=self.max_vals[mod].item())

            brain_slice = (brain_slice - self.means.reshape(-1, 1, 1)) / self.stds.reshape(-1, 1, 1)

            for mod in range(brain_slice.shape[0]):  # range(4)
                brain_slice[mod, :, :][brain_slice[mod, :, :] <= self.mins[mod]] = -1

            image_list.append(brain_slice.clone())

            missing_modality = random.randint(0, 3)
            missing_modality_list.append(torch.tensor(missing_modality))

            masked_slice = brain_slice.clone()
            masked_slice[missing_modality] = -1
            image_masked_list.append(masked_slice)

            missing_modality_slice = brain_slice[missing_modality, :, :].unsqueeze(0)
            target_list.append(missing_modality_slice)
        image = torch.stack(image_list, dim=0)
        image_masked = torch.stack(image_masked_list)
        target = torch.stack(target_list)
        segmentations = torch.stack(segmentations_list, dim=0)
        missing_modalities = torch.stack(missing_modality_list)
        return {
            'image': image,  # (B,4,240,240)
            'image_masked': image_masked,  # (B,4,240,240)
            'target': target,  # (B,1,240,240)
            'modality': missing_modalities,  # (B)
            'segmentations': segmentations  # (B,1,240,240)
        }


def brain_collate_stable_v1(batch):
    all_ids = []
    all_brains_tensor_list = []
    all_labels = []
    headers = []
    affines = []
    # all_affines = []
    # means = [1031.47412109375,752.4436645507812,463.6942138671875,624.90673828125]
    # stds = [1155.3835943096994,804.1193163952722,422.42170132463605,491.87728716113736]
    # upper_bounds = [4623, 3503, 1797, 2532]
    name_to_label_dict = {
        'seg.nii.gz': 0,
        't1c.nii.gz': 1,
        't1n.nii.gz': 2,
        't2f.nii.gz': 3,
        't2w.nii.gz': 4

    }
    for brains in batch:
        samebrain_tensor_list: List[torch.Tensor] = []  # list of all the modalities for the same brain
        samebrain_labels: List[torch.Tensor] = []  # list of
        all_ids.append(brains[
                           0].parent)  # getting the path of the directory of the specific patient. ex: /work_path/train_gli/BraTS-GLI-00000-000
        # this sections iterates over the dictionary, finding and processing the modalities in the right order
        samebrain_headers = []
        samebrain_affines = []
        for key, label in name_to_label_dict.items():

            brain = [elem for elem in brains if key in str(elem)]
            assert len(
                brain) <= 1, f'Dataloader found {len(brain)} brains in {all_ids[-1]}'  # always one or zero brain per modality
            if len(brain) == 0:  # missing modality
                pass
            else:
                samebrain_labels.append(torch.tensor(label))
                brain = brain[0]  # list to element
                brain_tensor, header, affine = get_tensor_header_affine_from_nii(brain)  # conversion to tensor
                # if label != 0: # do not normalize if it is the segmentation mask
                # brain_tensor = brain_tensor.clamp(min = 0, max=upper_bounds[label - 1])
                # brain_tensor = (brain_tensor - means[label-1])/stds[label-1]
                samebrain_tensor_list.append(brain_tensor)
                samebrain_headers.append(header)
                samebrain_affines.append(affine)
        samebrain_tensor = torch.stack(samebrain_tensor_list)  # stacking the modalities on a single tensor
        samebrain_labels_tensor = torch.stack(samebrain_labels)
        all_brains_tensor_list.append(samebrain_tensor)
        all_labels.append(samebrain_labels_tensor)

        headers.append(samebrain_headers)
        affines.append(samebrain_affines)
    all_brains_tensor = torch.stack(all_brains_tensor_list)
    all_labels_tensor = torch.stack(all_labels)
    return {
        'tensor': all_brains_tensor,
        'labels': all_labels_tensor,
        'ids': all_ids,
        'headers': headers,
        'affines': affines}


def brain_collate_3d_norm(batch):
    all_ids = []
    all_brains_tensor_list = []
    all_labels = []
    means = torch.tensor([1066.3375, 781.2247, 510.9852, 673.4393])
    stds = torch.tensor([1301.7006, 944.3418, 769.4159, 804.3864])
    max_vals = torch.tensor([8664, 7315, 8842, 8233])

    name_to_label_dict = {
        'seg.nii.gz': 0,
        't1c.nii.gz': 1,
        't1n.nii.gz': 2,
        't2f.nii.gz': 3,
        't2w.nii.gz': 4

    }
    for brains in batch:
        samebrain_tensor_list: List[torch.Tensor] = []  # list of all the modalities for the same brain
        samebrain_labels: List[torch.Tensor] = []  # list of
        all_ids.append(brains[
                           0].parent)  # getting the path of the directory of the specific patient. ex: /work_path/train_gli/BraTS-GLI-00000-000
        # this sections iterates over the dictionary, finding and processing the modalities in the right order
        for key, label in name_to_label_dict.items():
            brain = [elem for elem in brains if key in str(elem)]
            assert len(
                brain) <= 1, f'Dataloader found {len(brain)} brains in {all_ids[-1]}'  # always one or zero brain per modality
            if len(brain) == 0:  # missing modality
                pass
            else:
                samebrain_labels.append(torch.tensor(label))
                brain = brain[0]  # list to element
                brain_tensor, _, _ = get_tensor_header_affine_from_nii(brain)  # conversion to tensor
                if label != 0:  # do not normalize if it is the segmentation mask
                    brain_tensor = brain_tensor.clamp(min=0, max=max_vals[label - 1].item())
                    brain_tensor = (brain_tensor - means[label - 1]) / stds[label - 1]
                samebrain_tensor_list.append(brain_tensor)
        samebrain_tensor = torch.stack(samebrain_tensor_list)  # stacking the modalities on a single tensor
        samebrain_labels_tensor = torch.stack(samebrain_labels)
        all_brains_tensor_list.append(samebrain_tensor)
        all_labels.append(samebrain_labels_tensor)
    all_brains_tensor = torch.stack(all_brains_tensor_list)
    all_labels_tensor = torch.stack(all_labels)
    return {
        'image': all_brains_tensor[:, 1:].contiguous(),
        'segmentations': all_brains_tensor[:, 0:1].contiguous(),
        'labels': all_labels_tensor,
        'ids': all_ids
    }


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
            id)  # Getting the path of the directory of the specific patient. ex: /work_path/train_gli/BraTS-GLI-00000-000

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


def build_histograms():
    '''BraTS-GLI-01368-000-t1c.nii.gz
    BraTS-GLI-01368-000-t1n.nii.gz
    BraTS-GLI-01368-000-t2f.nii.gz
    BraTS-GLI-01368-000-t2w.nii.gz'''
    t1c_max = 2_120_538
    t1n_max = 155_724
    t2f_max = 612_368
    t2w_max = 4_563_634
    maxes = [t1c_max, t1n_max, t2f_max, t2w_max]

    # Initialize histograms
    histograms = [
        torch.zeros(t1c_max +1),
        torch.zeros(t1n_max +1),
        torch.zeros(t2f_max +1),
        torch.zeros(t2w_max +1)
    ]

    dataset = BrainDataset3D(dataset_dirs=['/work/tesi_ocarpentiero/train_gli',
                                         '/work/tesi_ocarpentiero/train_met',
                                         '/work/tesi_ocarpentiero/train_met_add'])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=brain_collate_stable_v1,
        shuffle=True,
        num_workers=8
    )

    for element in tqdm(dataloader):
        brains = element['tensor'].squeeze(0)
        for i in range(4):  # Corrected indexing: 0-3 instead of 1-4
            histograms[i] += torch.histc(brains[i+1],min = 0,max = maxes[i],bins=maxes[i]+1)

    print('end of iter')
    return histograms

