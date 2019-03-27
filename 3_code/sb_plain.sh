#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-04:00:00
#SBATCH --mem=65G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'
#SBATCH --output=./Max/sl-plain-%j.txt

# set -x # print all output to log file

# copy tfrecords files
cd /scratch/cs/ai_croppro/
mkdir /tmp/$SLURM_JOB_ID                                    # get a directory where you will send all output from your program
cp 2_data/03_images_subset_masked/*.tfrecord /tmp/$SLURM_JOB_ID    # copy tfrecords files to temporary directory
cp 2_data/03_images_subset_masked/*.txt /tmp/$SLURM_JOB_ID         # copy other necessary files to temporary directory

# load environment
module purge
module load anaconda3
source activate edward

which python

# run python script with temporary directory as input for the images
srun python 3_code/vae.py -c triton --data_path /tmp/$SLURM_JOB_ID/ -z 1024 --mse -e 70 --param_alternation plain --n_conv 10
