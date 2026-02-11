#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 8 \
              train.py --run-distill \
                       -p /path/to/embedding_alignment/checkpoint \
                       --distill-lr 2e-5 \
                       --distill-mbs 8192 \
                       --distill-steps 160000 \
                       -c /path/to/hnet_model/config \
                       --pretrain-transformer-name /path/to/pretrain_transformer/checkpoint \
                       --save-dir output/joint_distillation
