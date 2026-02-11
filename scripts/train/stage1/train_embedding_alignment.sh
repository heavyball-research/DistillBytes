#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 8 \
              train.py --run-embedding \
                       --embedding-lr 1e-3 \
                       --embedding-mbs 131072 \
                       --embedding-steps 10000 \
                       -c /path/to/hnet_model/config \
                       --pretrain-transformer-name /path/to/pretrain_transformer/checkpoint \
                       --save-dir output/embedding_alignment
