#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton

accelerate launch --num_processes 8 \
              train.py --run-routing \
                       -p /path/to/distill/checkpoint \
                       --routing-lr 1e-3 \
                       --routing-mbs 131072 \
                       --routing-steps 10000 \
                       -c /path/to/hnet_model/config \
                       --pretrain-transformer-name /path/to/pretrain_transformer/checkpoint \
                       --save-dir output/boundary_learning
