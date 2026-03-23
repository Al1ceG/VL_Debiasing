#!/bin/bash
#SBATCH --job-name=sfid_preprocessing
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --mail-user=s2142414@ed.ac.uk
#SBATCH --mail-type=END,FAIL

source ~/venv_mlp/bin/activate

cd ~/VL_Debiasing

./unified_debiasing/preprocessing/embedding_preprocessing.sh
