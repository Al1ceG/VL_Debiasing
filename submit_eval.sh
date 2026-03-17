#!/bin/bash
#SBATCH --job-name=clip_debias_eval
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --mail-user=s2142414@ed.ac.uk
#SBATCH --mail-type=END,FAIL

# Activate your virtual environment
source ~/venv_mlp/bin/activate

# Use local Java 8 for SPICE
export JAVA_HOME=$HOME/jdk8u402-b06
export PATH=$JAVA_HOME/bin:$PATH

# Create a larger temp directory
mkdir -p /home/s2142414/tmp

# Redirect temporary files away from /tmp
export TMPDIR=/home/s2142414/tmp
export TEMP=/home/s2142414/tmp
export TMP=/home/s2142414/tmp

# Verify Java version (helps debugging)
java -version

# Run your Python script
python -u VL_Debiasing/measure_caption_bias.py
