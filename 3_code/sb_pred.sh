#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-12:00:00
#SBATCH --mem=65G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'
#SBATCH --output=./Max/sl-predict-%j.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# load environment
module purge
module load anaconda3
source activate edward

which python

# run python script with temporary directory as input for the images
srun python 3_code/prediction.py -c triton -m 4_runs/logging/models/best_512x512_long_1024z_3Conv_0BN_80ep_MSE_2019-03-28_19-44-06
