#!/bin/bash
#SBATCH --job-name=tversky_madness
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -e /home/user/output/slurm/err_run_19.txt
#SBATCH -o /home/user/output/slurm/out_run_19.txt
#SBATCH --gres=gpu:4
#SBATCH --account=your_account
#SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G


source /path_to_venv/.venv/mri_crossmod_venv/bin/activate

accelerate launch --num_processes=4 --num_machines=1 --mixed_precision no --dynamo_backend no /work/your_account/cross-modality_synthesis/tesi_proj/src/training/train_hf_debugged.py \
                    --identifier run_19_twersky_loss  --batch-size 16   --epochs 10   --lr 0.00005 --ssim-coefficients 5 --num-workers 16 --resume \
                    --ssim-mode double --always-three --dice-coefficients 5 --compile --view sag \
                    --segmenter-weights /work_path/checkpoints/twersky_finetune_final_perversion_but_SAG/checkpoint_epoch_100.pth
 