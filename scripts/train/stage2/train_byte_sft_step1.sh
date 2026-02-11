#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 8 \
              train.py --run-dechunk \
                       -p /path/to/boundary_learning/checkpoint \
                       --dechunk-lr 1e-3 \
                       --dechunk-mbs 131072 \
                       --dechunk-steps 10000 \
                       -c /path/to/hnet_model/config \
                       --pretrain-transformer-name /path/to/pretrain_transformer/checkpoint \
                       --save-dir output/byte_sft_step1
