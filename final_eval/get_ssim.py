
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch
import numpy as np
import SimpleITK as sitk 
from pathlib import Path
from src.utils.preprocessing import get_tensor_header_affine_from_nii
from src.utils.indexing import conv_path_nnunet_to_brats, conv_path_nnunet_to_nnunet_segm
from argparse import ArgumentParser
from tqdm import tqdm

# Define evaluation Metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim = StructuralSimilarityIndexMeasure().to(device)

def __percentile_clip(input_tensor, reference_tensor=None, p_min=0.5, p_max=99.5, strictlyPositive=True):
    """Normalizes a tensor based on percentiles. Clips values below and above the percentile.
    Percentiles for normalization can come from another tensor.

    Args:
        input_tensor (torch.Tensor): Tensor to be normalized based on the data from the reference_tensor.
            If reference_tensor is None, the percentiles from this tensor will be used.
        reference_tensor (torch.Tensor, optional): The tensor used for obtaining the percentiles.
        p_min (float, optional): Lower end percentile. Defaults to 0.5.
        p_max (float, optional): Upper end percentile. Defaults to 99.5.
        strictlyPositive (bool, optional): Ensures that really all values are above 0 before normalization. Defaults to True.

    Returns:
        torch.Tensor: The input_tensor normalized based on the percentiles of the reference tensor.
    """
    if(reference_tensor == None):
        reference_tensor = input_tensor
    v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) #get p_min percentile and p_max percentile

    if( v_min < 0 and strictlyPositive): #set lower bound to be 0 if it would be below
        v_min = 0
    output_tensor = np.clip(input_tensor,v_min,v_max) #clip values to percentiles from reference_tensor
    output_tensor = (output_tensor - v_min)/(v_max-v_min) #normalizes values to [0;1]

    return output_tensor.cpu()

            
def compute_metrics(gt_image: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor, normalize=True):
    """Computes MSE, PSNR and SSIM between two images only in the masked region.

    Normalizes the two images to [0;1] based on the gt_image 0.5 and 99.5 percentile in the non-masked region.
    Requires input to have shape (1,1, X,Y,Z), meaning only one sample and one channel.
    For SSIM, we first separate the input volume to be tumor region and non-tumor region, then we apply regular SSIM on the complete volume. In the end we take
    the two volumes.

    Args:
        gt_image (torch.Tensor): The ground truth image (***.nii.gz)
        prediction (torch.Tensor): The inferred/predicted image
        mask (torch.Tensor): The segmentation mask (seg.nii.gz)
        normalize (bool): Normalizes the input by dividing trough the maximal value of the gt_image in the masked
            region. Defaults to True

    Raises:
        UserWarning: If you dimensions do not match the (torchmetrics) requirements: 1,1,X,Y,Z

    Returns:
        float: (SSIM_tumor, SSIM_non_tumor)
    """

    if not (prediction.shape[0] == 1 and prediction.shape[1] == 1):
        #raise UserWarning(f"All inputs have to be 5D with the first two dimensions being 1. Your prediction dimension: {prediction.shape}")
        gt_image = gt_image.unsqueeze(0).unsqueeze(0)
        prediction = prediction.unsqueeze(0).unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0)
    
    # Normalize to [0;1] individually after intensity clipping
    if normalize:
        gt_image = __percentile_clip(gt_image, p_min=0.5, p_max=99.5, strictlyPositive=True)
        prediction = __percentile_clip(prediction, p_min=0.5, p_max=99.5, strictlyPositive=True)
    mask[mask>0] = 1
    mask = mask.type(torch.int64)
    # Get Infill region (we really are only interested in the infill region)
    prediction_tumor = prediction * mask
    gt_image_tumor = gt_image * mask

    prediction_non_tumor = prediction * (1-mask)
    gt_image_non_tumor = gt_image * (1-mask)

 
    # SSIM - apply on complete masked image but only take values from masked region
    SSIM_tumor = ssim(preds=prediction_tumor.to(device), target=gt_image_tumor.to(device))
    SSIM_non_tumor = ssim(preds=prediction_non_tumor.to(device), target=gt_image_non_tumor.to(device))

    return float(SSIM_tumor), float(SSIM_non_tumor)


parser = ArgumentParser()
parser.add_argument('--id',type = str,required=True,help='Provide the base identifier of the checkpoint.\n'+
                    'Ex: if the checkpoint is at /work_path/checkpoints/final/run_14_always_three.safetensors,\n' +
                    'the id should be: --id run_14_always_three')
args = parser.parse_args()
id = args.id

# These paths only contain the reconstructed modality, in nnunet_convention
input_path_gli = Path(f'/work_path/brats3d/pseudo_random/recon_gli_{id}')
input_path_met = Path(f'/work_path/brats3d/pseudo_random/recon_met_{id}')

# These paths contain all the GT images 
reference_path_gli = Path('/work_path/brats3d/pseudo_random/complete_gli')
reference_path_met = Path('/work_path/brats3d/pseudo_random/complete_met')

# These paths contain all the precomputed segmentations
segm_reference_path_gli = Path('/work_path/brats3d/pseudo_random/gli_segmentations')
segm_reference_path_met = Path('/work_path/brats3d/pseudo_random/met_segmentations')

list_of_files = list(input_path_gli.iterdir()) + list(input_path_met.iterdir())
assert len(list_of_files) == 250, f'Got {len(list_of_files)} files instead of 250, so strange!'
met_ssims = [0.0,0.0]
gli_ssims = [0.0,0.0]
all_ssims = [0.0,0.0]
for file in tqdm(list_of_files):
    if 'MET' in str(file):
        base_path = reference_path_met
        base_segm_path = segm_reference_path_met
    elif 'GLI' in str(file):
        base_path = reference_path_gli
        base_segm_path = segm_reference_path_gli
    else:
        raise RuntimeError('Invalid type of tumor!')
    
    ref_volume_path = base_path / conv_path_nnunet_to_brats(file)
    ref_tensor, _,_ = get_tensor_header_affine_from_nii(ref_volume_path)

    recon_tensor, _,_ = get_tensor_header_affine_from_nii(file)

    segm_volume_path = base_segm_path / conv_path_nnunet_to_nnunet_segm(file)
    segm_tensor, _ ,_ = get_tensor_header_affine_from_nii(segm_volume_path)
    tum, non_tum = compute_metrics(ref_tensor,recon_tensor,segm_tensor)
    if not torch.isnan(torch.tensor(tum)):
        all_ssims[0]+=tum
    all_ssims[1]+=non_tum
    if 'MET' in str(file):
        met_ssims[0]+=tum
        met_ssims[1]+=non_tum
    elif 'GLI' in str(file):
        if not torch.isnan(torch.tensor(tum)):
            gli_ssims[0]+=tum
        gli_ssims[1]+=non_tum
    else:
        raise RuntimeError('Invalid type of tumor!')

met_ssims[0], met_ssims[1] = met_ssims[0]/len(list(input_path_met.iterdir())), met_ssims[1]/len(list(input_path_met.iterdir()))
gli_ssims[0], gli_ssims[1] = gli_ssims[0]/(len(list(input_path_gli.iterdir()))-11), gli_ssims[1]/len(list(input_path_gli.iterdir()))
all_ssims[0],all_ssims[1]= all_ssims[0]/(len(list_of_files) - 11),all_ssims[1]/len(list_of_files)
out_file = f'/home/user/results/{id}_ssim.txt'
with open(out_file, 'w') as f:
    print(f'ssims for id:{id}, in the order: tumor, non-tumor',file=f)
    print(f'total_ssims: {all_ssims}',file=f)
    print(f'gli ssims: {gli_ssims}',file=f)
    print(f'met_ssim: {met_ssims}',file=f)
