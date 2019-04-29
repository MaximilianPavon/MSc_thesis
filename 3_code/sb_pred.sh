#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-12:00:00
#SBATCH --mem=65G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'
#SBATCH --array=0-1
#SBATCH --output=./Max/sl-predict-%A_%a.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# load environment
module purge
module load anaconda3
source activate edward

which python

# run python script
if [ $SLURM_ARRAY_TASK_ID -eq 0 ];then
    srun python 3_code/prediction.py -c triton -m 4_runs/logging/models/in_field_loss_1024z_3Conv_0BN_40ep_MSE_0IFL_2019-04-10_16-45-46 -e 100 --param_alternation 0IFL
else
    srun python 3_code/prediction.py -c triton -m 4_runs/logging/models/in_field_loss_1024z_3Conv_0BN_40ep_MSE_1IFL_2019-04-10_16-45-57 -e 100 --param_alternation 1IFL
fi
