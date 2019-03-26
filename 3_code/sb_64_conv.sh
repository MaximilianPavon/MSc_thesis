#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-01:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --array=3-15
#SBATCH --output=./Max/sl-64_conv-%A_%a.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# load environment
module purge
module load anaconda3
source activate edward_cpu

which python

# run python script with temporary directory as input for the images
srun python 3_code/vae_64.py -c triton -z 1024 --mse -e 100 --param_alternation conv --n_conv $SLURM_ARRAY_TASK_ID
