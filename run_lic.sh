#!/bin/bash
#SBATCH --job-name=LIC_Eval
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/s2887183/VLD/VL_Debiasing/logs/lic_%j.out

# 1. Load your environment
. /home/htang2/toolchain-20251006/toolchain.rc
source ~/venv_mlp/bin/activate
cd ~/VLD/VL_Debiasing

# 2. Run the script
Make sure the path to the script is correct relative to VL_Debiasing
echo "Processing: BASELINE"
python -u unified_debiasing/compute_lic.py --file_path results/clip_cap_baseline.csv
echo "Processing: DEBIASED"
python -u unified_debiasing/compute_lic.py --file_path results/clipcap_debiased.csv

echo ">>> All Evaluations Finished at $(date)"

# # 2. Run the script for each file
# # IMPORTANT: Use double quotes around filenames with brackets/commas

# echo "Processing: DEBIASED SFID"
# python -u unified_debiasing/compute_lic.py --file_path "results/clipcap_debiased_sfid_debiased.csv"

# echo "Processing: DECODER ONLY"
# python -u unified_debiasing/compute_lic.py --file_path "results/clip_cap_sfid_['decoder']_50_50_0.9_features.csv"

# echo "Processing: ENCODER + DECODER"
# python -u unified_debiasing/compute_lic.py --file_path "results/clip_cap_sfid_['decoder', 'encoder']_50_50_0.9_features.csv"

# echo ">>> All Evaluations Finished at $(date)"