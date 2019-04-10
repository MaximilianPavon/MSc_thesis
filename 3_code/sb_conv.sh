#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-09:00:00
#SBATCH --mem=65G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'
#SBATCH --array=3-20
#SBATCH --output=./Max/sl-conv-%A_%a.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# load environment
module purge
module load anaconda3
source activate edward

which python

# run python script with temporary directory as input for the images
srun python 3_code/vae.py -c triton --mse -z 1024 -e 50 --param_alternation conv --n_conv $SLURM_ARRAY_TASK_ID
