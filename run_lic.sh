#!/bin/bash
#SBATCH --job-name=LIC_Eval
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s2142414/VL_Debiasing/logs/slurm-%j.out

# 1. Load your environment
source ~/venv_mlp/bin/activate
# Load GPU toolchain
. /home/htang2/toolchain-20251006/toolchain.rc
cd ~/VL_Debiasing

# 2. Run the script
# Make sure the path to the script is correct relative to VL_Debiasing
echo "Processing: BASELINE"
python -u unified_debiasing/compute_lic.py --file_path results/clip_cap_baseline.csv
echo "Processing: DEBIASED"
python -u unified_debiasing/compute_lic.py --file_path results/clipcap_debiased.csv

echo ">>> All Evaluations Finished at $(date)"

echo ">>> JOB COMPLETED AT: $(date)"