#!/bin/bash
#SBATCH --job-name=clip_debias_eval_new
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --mail-user=s2142414@ed.ac.uk
#SBATCH --mail-type=END,FAIL

source ~/venv_mlp/bin/activate

python -u VL_Debiasing/measure_caption_bias.py \
    --image_dir VL_Debiasing/data/COCO/images/val2014 \
    --results_filename /disk/scratch/s2142414/clip_cap_debiased.csv