#!/bin/bash
#SBATCH --job-name=tversky_madness
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -e /home/user/output/slurm/err_run_18.txt
#SBATCH -o /home/user/output/slurm/out_run_18.txt
#SBATCH --gres=gpu:4
#SBATCH --account=cvcs2024
#SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G


source /home/user/.venv/bin/activate

accelerate launch --num_processes=4 --num_machines=1 --mixed_precision no --dynamo_backend no /home/user/src/training/train_hf_debugged.py \
                    --identifier run_18_twersky_loss  --batch-size 16   --epochs 10   --lr 0.00005 --ssim-coefficients 5 --num-workers 16 --resume \
                    --ssim-mode double --always-three --dice-coefficients 5 --compile

sleep infinity