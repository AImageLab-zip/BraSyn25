#!/bin/bash
#SBATCH --job-name=brain_gen_run_24
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH -e /work/your_account/cross-modality_synthesis/output/slurm/err_run_24_2gpu.txt
#SBATCH -o /work/your_account/cross-modality_synthesis/output/slurm/out_run_24_2gpu.txt
#SBATCH --gres=gpu:2
#SBATCH --account=your_account
#SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G


source /path_to_venv/.venv/mri_crossmod_venv/bin/activate

export WANDB_ENTITY=wandb-account
export WANDB_PROJECT=HF_STUFF      

accelerate launch --main_process_port 29999 --num_processes=2 --num_machines=1 --mixed_precision no --dynamo_backend no /work/your_account/cross-modality_synthesis/tesi_proj/src/training/train_hf_debugged.py \
                    --identifier run_24_mask_grouped_all_bs16 --batch-size 16 --epochs 5 --lr 0.00005 --ssim-coefficients 5 --num-workers 16 \
                    --ssim-mode double --always-three --dice-coefficients 0 --compile --mask-background --view axi sag --dataset gli met met_add --grouped_enc

sleep 3000