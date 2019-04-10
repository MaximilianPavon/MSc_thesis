#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-08:00:00
#SBATCH --mem=85G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'
#SBATCH --array=7-10
#SBATCH --output=./Max/sl-long-%A_%a.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# load environment
module purge
module load anaconda3
source activate edward

which python

# define the dimensionality of latent space and pass it as argument to python script
z=$((2**$SLURM_ARRAY_TASK_ID))
echo "latent dim: $z"

# run python script with temporary directory as input for the images
srun python 3_code/vae.py -c triton -z $z --mse -e 80 --param_alternation long --n_conv 3
