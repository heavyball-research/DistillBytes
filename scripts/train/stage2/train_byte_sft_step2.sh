#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton

accelerate launch --num_processes 8 \
              train.py --run-dechunk-step2 \
                       -p /path/to/byte_sft_step1/checkpoint_model.pt \
                       --dechunk-step2-mbs 8192 \
                       --dechunk-step2-steps 160000 \
                       --dechunk-step2-lr 2e-5 \
                       --use-hierarchical-lrs \
                       -c /path/to/hnet_model/config \
                       --pretrain-transformer-name /path/to/pretrain_transformer/checkpoint \
                       --save-dir output/final_hnet_model
