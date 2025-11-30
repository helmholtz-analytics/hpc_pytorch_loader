#!/usr/bin/env bash

#activate virtual environemnt
source ../../dataloadenv/bin/activate

python caltech256_conversion.py \
    --input_path "/p/project1/hrfmri2/datasets/caltech256/" \
    --format "memmap" \
    --img_per_file 5000 \
    --batch_size 500 \
    --num_worker 12 \
    --output_dir "/p/scratch/hrfmri2/hmouda1/loader_demo/caltech256_memmap"
    

