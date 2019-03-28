#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-07:00:00
#SBATCH --mem=65G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'
#SBATCH --array=3-20
#SBATCH --output=./Max/sl-conv-%A_%a.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# copy tfrecords files to /tmp/$SLURM_ARRAY_JOB_ID
mkdir /tmp/$SLURM_ARRAY_JOB_ID 2> /dev/null   # try to create a directory for the array job, send error to /dev/null
response=$?                                   # catch the error code
if [[ ${response} -eq 0 ]];then
    # if the creation of the directory was succesfull:
    # copy files to temporary directory
    echo 'copying files...'
    start=`date +%s`
    cp 2_data/03_images_subset_masked/*.tfrecord /tmp/$SLURM_ARRAY_JOB_ID    # copy tfrecords files to temporary directory
    cp 2_data/03_images_subset_masked/*.txt /tmp/$SLURM_ARRAY_JOB_ID         # copy other necessary files to temporary directory
    end=`date +%s`
    runtime=$(($end-$start))
    echo "Copying took $runtime seconds"
else
    echo 'waiting 600s...'
    # if the creation of the directory was NOT succesfull (because it already exists):
    # wait until the data is copied to temporary directory
    sleep 600
fi

# load environment
module purge
module load anaconda3
source activate edward

which python

# run python script with temporary directory as input for the images
srun python 3_code/vae.py -c triton --data_path /tmp/$SLURM_ARRAY_JOB_ID/ --mse -z 1024 -e 70 --param_alternation conv --n_conv $SLURM_ARRAY_TASK_ID
