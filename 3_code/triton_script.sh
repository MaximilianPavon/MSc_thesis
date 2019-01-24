#!/bin/bash -l

#SBATCH --time=0-00:10:00
#SBATCH --mem-per-cpu 10G

#SBATCH --gres=gpu:2
#SBATCH --constraint='kepler|pascal|volta'

module load anaconda3/5.1.0-gpu
source activate edward

cd /scratch/cs/ai_croppro/

python 3_code/keras_vae.py -c triton