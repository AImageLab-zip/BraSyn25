#!/bin/bash
#SBATCH --job-name=brain_gen_run_25
#SBATCH --partition=boost_usr_prod 
#_SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=16:00:00
#SBATCH -e /home/user/output/slurm/err_25.txt
#SBATCH -o /home/user/output/slurm/out_25.txt
#SBATCH --gres=gpu:2
#SBATCH --account=your_account
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G


source /home/user/.venv/bin/activate



accelerate launch --main_process_port 29999 --num_processes=2 --num_machines=1 --mixed_precision no --dynamo_backend no /home/user/src/training/train_hf_debugged.py \
                    --identifier run_25_flash_views --batch-size 16 --epochs 5 --lr 0.00005 --ssim-coefficients 5 --num-workers 16 \
                    --ssim-mode double --always-three --dice-coefficients 0 --mask-background --view axi sag --dataset gli met met_add --grouped_enc \
                    --num-heads 16 \
                    --infuse-view --compile --resume
