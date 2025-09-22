#!/bin/bash
#SBATCH --job-name=name_of_your_job
#SBATCH --partition=partition_name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=2:00:00
#SBATCH -e error_file_path.txt
#SBATCH -o error_file_path.txt
#SBATCH --gres=gpu:1
#SBATCH --account=your_account

id=run_11_only_gli_no_ssim

source /path_to_venv/.venv/bin/activate
cd /home/user/final_eval
python run_generator.py --id $id --view sag --more _sag
python segment_gli.py --id ${id}_sag
python segment_met.py --id ${id}_sag
python get_ssim.py --id ${id}_sag
python get_dice_nsd_faster.py --id ${id}_sag 
