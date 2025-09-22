#!/bin/bash
#SBATCH --job-name=tversky_madness
#SBATCH --partition=all_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1:00:00
#SBATCH -e /home/user/output/slurm/debug.txt
#SBATCH -o /home/user/output/slurm/debug.txt
#SBATCH --gres=gpu:1
#SBATCH --account=your_account



source /home/user/.venv/bin/activate

accelerate launch --num_processes=1 --num_machines=1 --mixed_precision no --dynamo_backend no /home/user/src/training/train_hf_debugged.py \
                    --identifier debug_19  --batch-size 2   --epochs 10   --lr 0.00005 --ssim-coefficients 5 --num-workers 4 --resume \
                    --ssim-mode double --always-three --dice-coefficients 5 --view sag \
                    --segmenter-weights /work_path/checkpoints/twersky_finetune_final_perversion_but_SAG/checkpoint_epoch_100.pth
 