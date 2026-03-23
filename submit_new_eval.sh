#!/bin/bash
#SBATCH --job-name=run_eval
#SBATCH --partition=Teachingss
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/s2887183/VLD/VL_Debiasing/logs/lic_%j.out

# Environment setup
source ~/venv_mlp/bin/activate

# Java needed for METEOR
export JAVA_HOME=$HOME/jdk8u402-b06
export PATH=$JAVA_HOME/bin:$PATH
export TMPDIR=/home/s2142414/tmp
mkdir -p /home/s2887183/tmp
mkdir -p /home/s2887183/VLD/VL_Debiasing/logs

cd /home/s2887183/VLD/VL_Debiasing

# Run
echo ">>> Starting Evaluation at $(date)"
python3 run_evaluation.py
echo ">>> Evaluation Finished at $(date)"