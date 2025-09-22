from src.data.dataset import BrainDataset3D, brain_collate_stable_v1
from pathlib import Path
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def slice_the_dataset_now(base_dir: Path):
    gli_dataset = BrainDataset3D(dataset_dirs=base_dir)
    gli_dataloader = DataLoader(dataset=gli_dataset, batch_size=1, shuffle=False, collate_fn=brain_collate_stable_v1,
                                num_workers=20)
    views = ['axial']
    slice_max_list = [155, 240, 240]
    destination_dirs = {
        view: Path(base_dir.parent.parent / str('brats2d_' + view[0:3]) / base_dir.name) for view in views
    }
    slice_max = {
        view: max_val for view, max_val in zip(views, slice_max_list)
    }

    for gli_brain, labels, ids in tqdm(gli_dataloader):
        gli_brain = gli_brain.squeeze(0)
        for view in views:
            for slice_index in range(slice_max[view]):
                if view == 'axial':
                    brain_slice = gli_brain[0:5, :, :, slice_index]
                elif view == 'coronal':
                    brain_slice = gli_brain[0:5, :, slice_index, :]
                    brain_slice = F.pad(brain_slice, (42, 43))
                elif view == 'sagittal':
                    brain_slice = gli_brain[0:5, slice_index, :, :]
                    brain_slice = F.pad(brain_slice, (42, 43))
                else:
                    raise RuntimeError(f'Invalid view: {view}')

                if brain_slice[brain_slice != 0].numel() >= 10000:
                    nii_image = nib.Nifti1Image(brain_slice.cpu().numpy(), np.eye(4))
                    destination = destination_dirs[view] / f'{ids[0].name}_sl_{slice_index}.nii.gz'
                    nib.save(nii_image, destination)


if __name__ == '__main__':
    slice_the_dataset_now(Path('/work_path/brats3d/train_gli'))
    slice_the_dataset_now(Path('/work_path/brats3d/val_gli'))
    slice_the_dataset_now(Path('/work_path/brats3d/train_met'))
    slice_the_dataset_now(Path('/work_path/brats3d/train_met_add'))
    slice_the_dataset_now(Path('/work_path/brats3d/val_met'))