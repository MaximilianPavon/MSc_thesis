#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-01:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --array=1-10
#SBATCH --output=./Max/sl-64_latent_dim-%A_%a.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# load environment
module purge
module load anaconda3
source activate edward_cpu

which python

# define the dimensionality of latent space and pass it as argument to python script
z=$((2**$SLURM_ARRAY_TASK_ID))
echo "latent dim: $z"

# run python script with temporary directory as input for the images
srun python 3_code/vae_64.py -c triton -z $z --mse -e 100 --param_alternation latent_dim

