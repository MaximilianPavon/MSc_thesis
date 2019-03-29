#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-02:00:00
#SBATCH --mem=65G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta'
#SBATCH --output=./Max/sl-predict-%j.txt

# set -x # print all output to log file

cd /scratch/cs/ai_croppro/

# copy tfrecords files
start=`date +%s`
mkdir /tmp/$SLURM_JOB_ID                                    # get a directory where you will send all output from your program
cp 2_data/03_images_subset_masked/*.tfrecord /tmp/$SLURM_JOB_ID    # copy tfrecords files to temporary directory
cp 2_data/03_images_subset_masked/*.txt /tmp/$SLURM_JOB_ID         # copy other necessary files to temporary directory
end=`date +%s`
runtime=$(($end-$start))
echo "Copying took $runtime seconds"

# load environment
module purge
module load anaconda3
source activate edward

which python

# run python script with temporary directory as input for the images
srun python 3_code/prediction.py -c triton --data_path /tmp/$SLURM_JOB_ID/ -m 4_runs/logging/models/best_512x512_long_1024z_3Conv_0BN_80ep_MSE_2019-03-28_19-44-06
