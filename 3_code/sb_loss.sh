#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-04:00:00
#SBATCH --mem=65G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'
#SBATCH --array=0-1
#SBATCH --output=sl-loss-%A_%a.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# copy tfrecords files to /tmp/$SLURM_ARRAY_JOB_ID
mkdir /tmp/$SLURM_ARRAY_JOB_ID 2> /dev/null   # try to create a directory for the array job, send error to /dev/null
response=$?                                   # catch the error code
if [[ ${response} -eq 0 ]];then
    # if the creation of the directory was succesfull:
    # copy files to temporary directory
    echo 'copying files...'
    cp 2_data/05_images_masked/*.tfrecord /tmp/$SLURM_ARRAY_JOB_ID    # copy tfrecords files to temporary directory
    cp 2_data/05_images_masked/*.txt /tmp/$SLURM_ARRAY_JOB_ID         # copy other necessary files to temporary directory
else
    echo 'waiting 240s...'
    # if the creation of the directory was NOT succesfull (because it already exists):
    # wait until the data is copied to temporary directory
    sleep 240
fi

# load environment
module purge
module load anaconda3
source activate edward

which python

if [ $SLURM_ARRAY_TASK_ID -eq 0 ];then
    srun python 3_code/vae.py -c triton --data_path /tmp/$SLURM_ARRAY_JOB_ID/ -z 128 --batch_normalization --param_alternation loss
else
    srun python 3_code/vae.py -c triton --data_path /tmp/$SLURM_ARRAY_JOB_ID/ -z 128 --batch_normalization --param_alternation loss --mse
fi
