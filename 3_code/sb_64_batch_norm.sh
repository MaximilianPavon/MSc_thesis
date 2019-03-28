#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-01:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --array=0-1
#SBATCH --output=./Max/sl-64_batch_norm_fixed-%A_%a.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# load environment
module purge
module load anaconda3
source activate edward_cpu

which python

prefix=batch_norm

if [ $SLURM_ARRAY_TASK_ID -eq 0 ];then
    srun python 3_code/vae_64.py -c triton -z 1024 --mse -e 100 --n_conv 3 --param_alternation ${prefix}
else
    srun python 3_code/vae_64.py -c triton -z 1024 --mse -e 100 --n_conv 3 --param_alternation ${prefix} --batch_normalization
fi
