#!/bin/bash
#SBATCH --job-name=2d_segmentation_abomination_definitely_last_one_DELUXE_MET_DEBUG
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH -e /home/user/output/slurm/err_segment_fusion_axi.txt
#SBATCH -o /home/user/output/slurm/out_segment_fusion_axi.txt
#SBATCH --gres=gpu:1
#SBATCH --account=your_account
################àSBATCH --constraint='gpu_RTX6000_24G|gpu_RTXA5000_24G'
#SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G

source /home/user/.venv/bin/activate
export PYTHONPATH=/home/user:$PYTHONPATH
python /home/user/src/training/train_segmenter_fusion.py --identifier segment_fusion_axi \
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
/work_path/brats2d_axi/train_met_tumoral \
/work_path/brats2d_axi/train_met_add_tumoral \
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
