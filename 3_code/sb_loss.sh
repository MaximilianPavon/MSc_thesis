#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-04:00:00
#SBATCH --mem=65G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'
#SBATCH --array=0-1

# set -x # print all output to log file

# load environment
module purge
module load anaconda3
source activate edward

which python
# env

# copy tfrecords files
cd /scratch/cs/ai_croppro/
mkdir /tmp/$SLURM_JOB_ID                                    # get a directory where you will send all output from your program
cp 2_data/05_images_masked/*.tfrecord /tmp/$SLURM_JOB_ID    # copy tfrecords files to temporary directory
cp 2_data/05_images_masked/*.txt /tmp/$SLURM_JOB_ID         # copy other necessary files to temporary directory
                                                            # run python script with temporary directory as input for the images

if [ $SLURM_ARRAY_TASK_ID -eq 0 ];then
    srun python 3_code/tf_vae_tfrecords.py -c triton --data_path /tmp/$SLURM_JOB_ID/ -z 128 --batch_normalization -e 125
else
    srun python 3_code/tf_vae_tfrecords.py -c triton --data_path /tmp/$SLURM_JOB_ID/ -z 128 --batch_normalization -e 125 --mse
fi
