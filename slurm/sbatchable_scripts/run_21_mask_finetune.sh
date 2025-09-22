#!/bin/bash
#SBATCH --job-name=tversky+masked_ssim+cycle+mask_bg+total_crazyness
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH -e /home/user/output/slurm/err_run_21.txt
#SBATCH -o /home/user/output/slurm/out_run_21.txt
#SBATCH --gres=gpu:2
#SBATCH --account=your_account
#SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G


source /home/user/.venv/bin/activate

accelerate launch --num_processes=2 --num_machines=1 --mixed_precision no --dynamo_backend no --main_process_port 29994 /home/user/src/training/train_hf_debugged.py \
                    --identifier run_21_mask_finetune  --batch-size 16   --epochs 3   --lr 0.00005 --ssim-coefficients 5 --num-workers 16 --resume \
                    --ssim-mode double --always-three --dice-coefficients 5 --compile --mask-background \
                    --pretrain-weights /work_path/checkpoints/hf_stuff/run_14_always_three

sleep infinity