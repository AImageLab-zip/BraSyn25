import shutil
import os

id = 'run_id' # Use the checkpoint name without the .safetensors at the end
output_path = f'/work_path/pseudo_random/gli_segm_{id}'
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)
shutil.copytree(f'/work_path/tmp/inference_{id}/ensemble/GB_GL_GS_RB_RL_RS_rGB_rGL_rGS/WT250_TC150_ET100/',
                output_path,dirs_exist_ok=True)