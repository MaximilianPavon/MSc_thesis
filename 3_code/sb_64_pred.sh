#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-01:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --output=./Max/sl-64_predict-%j.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# load environment
module purge
module load anaconda3
source activate edward_cpu

which python

# run python script with temporary directory as input for the images
srun python 3_code/prediction_64.py -c triton -m 4_runs/logging/models/best_64x64_long_1024z_3Conv_0BN_200ep_MSE_2019-03-28_11-21-28
