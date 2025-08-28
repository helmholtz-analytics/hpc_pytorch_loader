#!/usr/bin/env bash

#activate virtual environemnt
source ../../dataloadenv/bin/activate

python caltech256_conversion.py \
    --input_path "." \
    --format "memmap" \
    --img_per_file 5000 \
    --batch_size 500 \
    --num_worker 4
    --output_path "caltech256_memmap" \
    

