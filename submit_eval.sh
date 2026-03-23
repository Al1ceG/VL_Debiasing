#!/bin/bash
#SBATCH --job-name=caption_eval
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2887183/VLD/VL_Debiasing/logs/lic_%j.out

# Ensure directories exist
mkdir -p /home/s2142414/logs
mkdir -p /home/s2142414/tmp

# Environment setup
source ~/venv_mlp/bin/activate

# THE "NECESSARY" JAVA STUFF
export JAVA_HOME=$HOME/jdk8u402-b06
export PATH=$JAVA_HOME/bin:$PATH
export TMPDIR=/home/s2142414/tmp

# Run
cd ~/VL_Debiasing
python -u measure_caption_bias.py