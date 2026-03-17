#!/bin/bash
#SBATCH --job-name=caption_gen_debiased
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2142414/VL_Debiasing/logs/slurm-%j.out


mkdir -p /home/s2142414/VL_Debiasing/results_debiased
mkdir -p /home/s2142414/VL_Debiasing/logs
mkdir -p /home/s2142414/tmp

source ~/venv_mlp/bin/activate

export JAVA_HOME=$HOME/jdk8u402-b06
export PATH=$JAVA_HOME/bin:$PATH
export TMPDIR=/home/s2142414/tmp

cd ~/VL_Debiasing

python -u measure_caption_bias.py \
    --image_dir data/COCO/images/val2014 \
    --results_filename results_debiased/clipcap_debiased.csv

