#!/bin/bash
#SBATCH --job-name=run_eval
#SBATCH --partition=Teaching
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2142414/VL_Debiasing/logs/slurm-%j.out

# Environment setup
source ~/venv_mlp/bin/activate

# Java needed for METEOR
export JAVA_HOME=$HOME/jdk8u402-b06
export PATH=$JAVA_HOME/bin:$PATH
export TMPDIR=/home/s2142414/tmp
mkdir -p /home/s2142414/tmp
mkdir -p /home/s2142414/VL_Debiasing/logs
mkdir -p ~/VL_Debiasing/results

# Run
# Give Java 8GB of RAM to handle SPICE scene graphs
export _JAVA_OPTIONS="-Xmx8g"

cd ~/VL_Debiasing
python run_evaluation.py