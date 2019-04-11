#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-02:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --array=0-1
#SBATCH --output=./Max/sl-64_predict-%A_%a.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# load environment
module purge
module load anaconda3
source activate edward_cpu

which python

# run python script


# run python script
if [ $SLURM_ARRAY_TASK_ID -eq 0 ];then
    srun python 3_code/prediction_64.py -c triton -m 4_runs/logging/models/64x64_long_1024z_3Conv_0BN_200ep_MSE_0IFL_2019-04-10_14-23-56/ --param_alternation 0IFL
else
    srun python 3_code/prediction_64.py -c triton -m 4_runs/logging/models/64x64_long_1024z_3Conv_0BN_200ep_MSE_1IFL_2019-04-10_15-42-06/ --param_alternation 1IFL
fi
