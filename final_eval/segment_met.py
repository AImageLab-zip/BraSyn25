import os, time
import numpy as np
import glob


import numpy as np
import torch
import nibabel as nib

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import SlidingWindowInfererAdapt
from monai.transforms import Compose, LoadImaged, NormalizeIntensityd, CropForegroundd, Invertd, EnsureTyped

from monai.networks.nets.segresnet_ds import SegResNetDS
from monai.utils import ImageMetaKey, convert_to_dst_type
from torch.cuda.amp import autocast

from pathlib import Path
from argparse import ArgumentParser
import shutil

DEBUG = False
NUM_WORKERS = 8

import re
def sorted_alphanumeric( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def get_validation_files(data_path:str):

    cases = glob.glob(data_path+'/*')
    cases = sorted_alphanumeric(cases)

    print('get_validation_files cases', cases)

    validation_files = []
    for case in cases:
        basename = os.path.basename(case)
        d = [os.path.join(case, basename+'-t2f.nii.gz'),
             os.path.join(case, basename+'-t1c.nii.gz'),
             os.path.join(case, basename+'-t1n.nii.gz'),
             os.path.join(case, basename+'-t2w.nii.gz')]
        validation_files.append({'image' : d})

    # print('validation_files', validation_files)
    return validation_files

# def post_processing(seg, filename=None):

#     et = (seg==3)
#     if np.count_nonzero(et) < 100:
#         seg[et] = 1
#         print('Removing small et', np.count_nonzero(et), filename)

#     return seg


def load_checkpoint(checkpoint):


    starttime = time.time()
    ckpt = torch.load(checkpoint, map_location=torch.device('cpu'))
    config = ckpt['config']
    network = config['network']    
    model = SegResNetDS(**network) 
    model.load_state_dict(ckpt['state_dict'], strict=True)

    model = model.to(device=torch.device('cuda'), memory_format=torch.channels_last_3d)
    model.eval()
    if DEBUG: print('checkpoint loaded ', checkpoint,  "time {:.2f}s".format(time.time() - starttime), 'epoch', ckpt['config']['network'], ckpt['epoch'], 'best_metric', ckpt['best_metric'] )

    return model, config


def infer_checkpoints(data, checkpoints, keep_head_channel):

    preds = []

    for checkpoint in checkpoints:
        model, config = load_checkpoint(checkpoint=checkpoint)

        sliding_inferrer = SlidingWindowInfererAdapt(roi_size=config['roi_size'], sw_batch_size=1, overlap=0.625, mode="gaussian", cache_roi_weight_map=False, progress=False)
       
        b_clip = (config['normalize_mode']=='meanstdtanh')
        image = 3 * torch.tanh(data / 3) if b_clip else data

        with autocast(enabled=True):
            pred = sliding_inferrer(inputs=image, network=model).float()
        pred = torch.sigmoid(pred, out=pred)
        preds.append(pred)
        pred = None
        image = None

    return preds

def ensemble(preds, enmax = 7):

    if len(preds)>enmax:
        vals_all = torch.zeros(len(preds))
        for i in range(len(preds)):
            par = preds[i][preds[i] >= 0.5]
            vals_all[i] = torch.mean(par).cpu() if par.numel()>0 else torch.tensor(0.)
        vals = list(torch.argsort(vals_all, descending=True)[:enmax].numpy())
        # print('vals sorting', vals_all.numpy(), vals)
        pred = sum([preds[v] for v in vals])/len(vals)
    else:
        pred = sum(preds)/len(preds)
    
    return pred

def process(data_path:str, output_path:str):

    print('START: data_path is', data_path, 'output_path', output_path)
    validation_files = get_validation_files(data_path)

    load_transforms = Compose([
        LoadImaged('image', ensure_channel_first=True, dtype=None, image_only=True),
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float),
        CropForegroundd(keys=["image"], source_key="image", margin=10, allow_smaller=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])
            
    inverse_transform = Invertd(keys="pred", orig_keys="image", transform=load_transforms, nearest_interp=False)

    ds = Dataset(data = validation_files, transform=load_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS) 

    device = torch.device('cuda')
    checkpoints_wt=['model_wt_0.pt', 'model_wt_1.pt', 'model_wt_2.pt', 'model_wt_3.pt', 'model_wt_4.pt', 'model_wt_5.pt', 'model_wt_6.pt', 'model_wt_7.pt', 'model_wt_8.pt', 'model_wt_9.pt', 'model_wt_10.pt', 'model_wt_11.pt', 'model_wt_12.pt', 'model_wt_13.pt', 'model_wt_14.pt']
    checkpoints_tc=['model_tc_0.pt', 'model_tc_1.pt', 'model_tc_2.pt', 'model_tc_3.pt', 'model_tc_4.pt', 'model_tc_5.pt', 'model_tc_6.pt', 'model_tc_7.pt', 'model_tc_8.pt', 'model_tc_9.pt', 'model_tc_10.pt', 'model_tc_11.pt', 'model_tc_12.pt', 'model_tc_13.pt', 'model_tc_14.pt']
    checkpoints_et=['model_et_0.pt', 'model_et_1.pt', 'model_et_2.pt', 'model_et_3.pt', 'model_et_4.pt', 'model_et_5.pt', 'model_et_6.pt', 'model_et_7.pt', 'model_et_8.pt', 'model_et_9.pt', 'model_et_10.pt', 'model_et_11.pt', 'model_et_12.pt', 'model_et_13.pt', 'model_et_14.pt']
    checkpoints_root='/work_path/met_segm_chk'

    for c in range(len(checkpoints_wt)): checkpoints_wt[c] = os.path.join(checkpoints_root, checkpoints_wt[c] )
    for c in range(len(checkpoints_tc)): checkpoints_tc[c] = os.path.join(checkpoints_root, checkpoints_tc[c] )
    for c in range(len(checkpoints_et)): checkpoints_et[c] = os.path.join(checkpoints_root, checkpoints_et[c] )
    

    with torch.no_grad():
        for batch_data in loader:
            
            loader_time = time.time()

            filename = batch_data['image'].meta[ImageMetaKey.FILENAME_OR_OBJ][0]
            casename = filename.split('/')[-2]
            filename_out = os.path.join(output_path, casename+'.nii.gz')
            
            data = batch_data["image"].as_subclass(torch.Tensor).to(device=device, memory_format=torch.channels_last_3d)
            print("Processing", casename, 'data', data.shape)

            starttime_loop = time.time()
            preds = infer_checkpoints(data=data, checkpoints=checkpoints_wt, keep_head_channel=0)
            print("Preds WT", preds[0].shape, len(preds), filename, "time {:.2f}s".format(time.time() - starttime_loop))
            pred_wt = ensemble(preds)
            preds = None

            starttime_loop = time.time()
            preds = infer_checkpoints(data=data, checkpoints=checkpoints_tc, keep_head_channel=1)
            print("Preds TC", preds[0].shape, len(preds), filename, "time {:.2f}s".format(time.time() - starttime_loop))
            pred_tc = ensemble(preds)
            preds = None


            starttime_loop = time.time()
            preds = infer_checkpoints(data=data, checkpoints=checkpoints_et, keep_head_channel=2)
            print("Preds ET", preds[0].shape, len(preds),  filename, "time {:.2f}s".format(time.time() - starttime_loop))
            pred_et = ensemble(preds)
            preds = None

            data = None

            pred = torch.cat((pred_wt, pred_tc, pred_et), dim=1)
            pred = pred.cpu()
            batch_data["pred"] = convert_to_dst_type(pred, batch_data["image"], dtype=pred.dtype, device=pred.device)[0]

            for b in decollate_batch(batch_data):
                    
                b = inverse_transform(b)
                pred = b.pop("pred")
                if DEBUG: print('unpadded pred shape', pred.shape)

                #convert to brats format
                pred_bin = pred >= 0.5
        
                pred = 2 * pred_bin.any(0).to(dtype=torch.uint8)
                pred[pred_bin[1:].any(0)] = 1
                pred[pred_bin[2]] = 3

                seg = pred.cpu().numpy().astype(np.uint8)
                pred = None

                # seg = post_processing(seg, filename=filename)

                image_input = nib.load(filename)
                seg = nib.Nifti1Image(seg, affine=image_input.affine)
                nib.save(seg, filename=filename_out)
                print('Done, image saved to ', filename_out,  seg.shape, "time {:.2f}s".format(time.time() - loader_time))






if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--id',type = str,required=True,help='Provide the base identifier of the checkpoint.\n'+
                        'Ex: if the checkpoint is at /work_path/checkpoints/final/run_14_always_three.safetensors,\n' +
                        'the id should be: --id run_14_always_three')
    args = parser.parse_args()
    id = args.id

    data_path = f'/work_path/brats3d/pseudo_random/complete_recon_met_{id}'
    output_path = f'/work_path/brats3d/pseudo_random/met_segm_{id}'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    process(data_path=data_path, output_path=output_path)