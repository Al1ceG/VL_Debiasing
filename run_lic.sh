#!/bin/bash
#SBATCH --job-name=compute_lic
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G

# Activate your python environment here if you use one (e.g., conda activate myenv)
source ~/venv_mlp/bin/activate
cd ~/VL_Debiasing


echo "============================================="
echo "Running LIC for Baseline Captions..."
echo "============================================="
python unified_debiasing/compute_lic.py --file_path results/clip_cap_baseline.csv --batch_size 64

echo "============================================="
echo "Running LIC for Debiased Captions..."
echo "============================================="
python unified_debiasing/compute_lic.py --file_path results/clipcap_debiased.csv --batch_size 64