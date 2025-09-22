#!/bin/bash
#SBATCH --job-name=gli_gen_but_without_stupid_bugs
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH -e /home/user/output/slurm/err_run_11.txt
#SBATCH -o /home/user/output/slurm/out_run_11.txt
#SBATCH --gres=gpu:2
#SBATCH --account=your_account



source /home/user/.venv/bin/activate

accelerate launch --num_processes=2 --num_machines=1 --mixed_precision fp16 --dynamo_backend no /home/user/src/training/train_hf_debugged.py \
                    --identifier run_11_only_gli_no_ssim  --batch-size 32   --epochs 10   --lr 0.00005 --ssim-coefficients 0 --num-workers 16 --resume

