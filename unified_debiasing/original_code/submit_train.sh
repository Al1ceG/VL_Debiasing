#!/bin/bash
#SBATCH --job-name=clip_debias
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --mail-user=s2345314@ed.ac.uk
#SBATCH --mail-type=END,FAIL

# Activate your virtual environment
source ~/venv_mlp/bin/activate

# Run your Python script
python -u train.py
