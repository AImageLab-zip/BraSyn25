#!/bin/bash
#SBATCH --job-name=OH_YES_2d_segmentation_abomination_waka_waka_eh_eh
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH -e /home/user/output/slurm/err_segment_fusion.txt
#SBATCH -o /home/user/output/slurm/out_segment_fusion.txt
#SBATCH --gres=gpu:1
#SBATCH --account=your_account
#SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G

source /home/user/.venv/bin/activate
export PYTHONPATH=/home/user:$PYTHONPATH
python /home/user/src/training/train_segmenter_total_fusion.py --identifier segmenter_ultimate_DELUXE_final_swag \
--start-check-index 100 \
\
--pretrain-datasets \
/work_path/brats2d_sag/train_met_add_tumoral \
/work_path/brats2d_sag/train_met_tumoral \
/work_path/brats2d_sag/train_gli_tumoral \
\
/work_path/brats2d_cor/train_met_add_tumoral \
/work_path/brats2d_cor/train_met_tumoral \
/work_path/brats2d_cor/train_gli_tumoral \
\
/work_path/brats2d_axi/train_met_add_tumoral \
/work_path/brats2d_axi/train_met_tumoral \
/work_path/brats2d_axi/train_gli_tumoral \
\
\
--batch-size 80 \
--num-classes 4 \
--pretrain-fraction 0.3 \
\
--finetune-datasets \
/work_path/brats2d_sag/train_met_add_tumoral \
/work_path/brats2d_sag/train_met_tumoral \
/work_path/brats2d_sag/train_gli_tumoral \
\
/work_path/brats2d_cor/train_met_add_tumoral \
/work_path/brats2d_cor/train_met_tumoral \
/work_path/brats2d_cor/train_gli_tumoral \
\
/work_path/brats2d_axi/train_met_add_tumoral \
/work_path/brats2d_axi/train_met_tumoral \
/work_path/brats2d_axi/train_gli_tumoral \
\
--val-datasets \
/work_path/brats2d_axi/train_met_tumoral \
/work_path/brats2d_axi/train_met_add_tumoral \
/work_path/brats2d_axi/train_gli_tumoral \
\
--train-splits \
/work_path/split_files/met_add_val.csv \
/work_path/split_files/met_add_train.csv \
/work_path/split_files/met_val.csv \
/work_path/split_files/met_train.csv \
/work_path/split_files/gli_val.csv \
/work_path/split_files/gli_train.csv \
\
--val-splits \
/work_path/split_files/met_add_test.csv \
/work_path/split_files/met_test.csv \
/work_path/split_files/gli_test.csv
