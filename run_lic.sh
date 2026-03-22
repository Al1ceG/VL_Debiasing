#!/bin/bash
#SBATCH --job-name=compute_lic
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/s2142414/VL_Debiasing/logs/slurm-%j.out


# Activate your python environment here if you use one (e.g., conda activate myenv)
source ~/venv_mlp/bin/activate
cd ~/VL_Debiasing

echo "============================================="
echo "Running LIC for Baseline Captions..."
echo "============================================="
python unified_debiasing/compute_lic.py --file_path results/clip_cap_baseline.csv

echo "============================================="
echo "Running LIC for Debiased Captions..."
echo "============================================="
python unified_debiasing/compute_lic.py --file_path results/clipcap_debiased.csv

echo "Job completed at $(date)"