#!/bin/bash -l

# request resources
# -----------------------
#SBATCH --time=0-1:00:00
#SBATCH --mem-per-cpu 32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint='kepler|pascal|volta'

# load environment
module purge
module load anaconda3
source activate edward

which python

cd /scratch/cs/ai_croppro/
mkdir /tmp/$SLURM_JOB_ID                                # get a directory where you will send all output from your program
cp 2_data/05_images_masked.tar.gz /tmp/$SLURM_JOB_ID    # copy zipped images to temporary directory
cd /tmp/$SLURM_JOB_ID                                   # go to new temporary directory and
tar xzf 05_images_masked.tar.gz                         # unzip images in temporary directory

cd /scratch/cs/ai_croppro/                              # go back to project folder and run python script with
                                                        # - temporary directory as input for the images
                                                        # - previously created model as input

for i in `seq 1 30`;                                    # create 30 different plots
        do
            echo $i
            python 3_code/keras_vae.py -c triton --data_path /tmp/$SLURM_JOB_ID/05_images_masked/ -m 4_runs/logging/checkpoints/6n_Conv_2019-01-24_19-43-28_0095-55149.27.hdf5
        done
