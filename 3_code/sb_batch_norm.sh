#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-07:00:00
#SBATCH --mem=65G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'
#SBATCH --array=0-1
#SBATCH --output=./Max/sl-batch_norm-%A_%a.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# load environment
module purge
module load anaconda3
source activate edward

which python

prefix=batch_norm

if [ $SLURM_ARRAY_TASK_ID -eq 0 ];then
    srun python 3_code/vae.py -c triton -z 1024 --n_conv 3 --mse --param_alternation ${prefix} -e 40
else
    srun python 3_code/vae.py -c triton -z 1024 --n_conv 3 --mse --param_alternation ${prefix} -e 40 --batch_normalization
fi
