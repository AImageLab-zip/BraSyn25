#!/bin/bash
#SBATCH --job-name=brain_gen_run_22
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH -e /work/your_account/cross-modality_synthesis/output/slurm/err_run_22.txt
#SBATCH -o /work/your_account/cross-modality_synthesis/output/slurm/out_run_22.txt
#SBATCH --gres=gpu:2
#SBATCH --account=your_account
#SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G


source /path_to_venv/.venv/mri_crossmod_venv/bin/activate

accelerate launch --main_process_port 29999 --num_processes=2 --num_machines=1 --mixed_precision no --dynamo_backend no /work/your_account/cross-modality_synthesis/tesi_proj/src/training/train_hf_debugged.py \
                    --identifier run_22_mask_finetune  --batch-size 16   --epochs 3   --lr 0.00005 --ssim-coefficients 5 --num-workers 16 --resume \
                    --ssim-mode double --always-three --dice-coefficients 5 --compile --mask-background --pretrain-weights /work_path/checkpoints/hf_stuff/run_14_always_three