#!/bin/bash
#SBATCH --job-name=qual_eval
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/s2887183/VLD/VL_Debiasing/logs/compare_%j.out

# 1. Environment setup
source /home/s2887183/venv_mlp/bin/activate

# 2. Move to the project root
cd /home/s2887183/VLD/VL_Debiasing

# 3. Run the script
# Replace 'find_examples.py' with whatever you named your python file
echo ">>> Starting Qualitative Comparison..."
python3 check_captions.py
echo ">>> Done. Check the results folder for CSVs."