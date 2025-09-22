#!/bin/bash
#SBATCH --job-name=brain_gen_run_24
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH -e /work/your_account/cross-modality_synthesis/output/slurm/err_run_24.txt
#SBATCH -o /work/your_account/cross-modality_synthesis/output/slurm/out_run_24.txt
#SBATCH --gres=gpu:4
#SBATCH --account=your_account
#SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G


source /path_to_venv/.venv/mri_crossmod_venv/bin/activate

export WANDB_ENTITY=wandb-account
export WANDB_PROJECT=HF_STUFF      

accelerate launch --main_process_port 29999 --num_processes=4 --num_machines=1 --mixed_precision no --dynamo_backend no /work/your_account/cross-modality_synthesis/tesi_proj/src/training/train_hf_debugged.py \
                    --identifier run_24_mask_grouped_all_bs24 --batch-size 24 --epochs 5 --lr 0.00005 --ssim-coefficients 5 --num-workers 16 \
                    --ssim-mode double --always-three --dice-coefficients 0 --compile --mask-background --view axi sag --dataset gli met met_add --grouped_enc

accelerate launch --main_process_port 29999 --num_processes=4 --num_machines=1 --mixed_precision no --dynamo_backend no /work/your_account/cross-modality_synthesis/tesi_proj/src/training/train_hf_debugged.py \
                    --identifier run_24_mask_grouped_all_bs20 --batch-size 20 --epochs 5 --lr 0.00005 --ssim-coefficients 5 --num-workers 16 \
                    --ssim-mode double --always-three --dice-coefficients 0 --compile --mask-background --view axi sag --dataset gli met met_add --grouped_enc

sleep 3000