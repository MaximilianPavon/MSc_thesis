#!/usr/bin/env bash

. /anaconda3/etc/profile.d/conda.sh
conda activate
conda activate edward

N_MAX=20

cd ~/Dropbox\ \(Aalto\)/3_code
i=1
for file in ../2_data/03_data/dataset1/*.tiff; do
#    echo $file
#    echo $i
    python tiff2rgb.py -f ${file}
    i=$((i+1))
    if [[ $i -gt N_MAX ]]; then
        break
    fi
done
