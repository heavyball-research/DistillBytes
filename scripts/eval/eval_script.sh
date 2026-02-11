#!/bin/bash
CHECKPOINT=/path/to/hnet_model/checkpoint
CONFIG=/path/to/hnet_model/config
MODEL_PATH=/path/to/pretrain_transformer/checkpoint
MODE=(stage1 or stage2)

accelerate launch --num_processes 8 eval.py --tasks hellaswag,lambada_openai,arc_easy,arc_challenge,piqa,winogrande,openbookqa \
    --model hnet \
    --model_args checkpoint_path=$CHECKPOINT,config_path=$CONFIG,pretrain_transformer_name=$MODEL_PATH,device=cuda,custom_batch_size=64,max_length=8192,mode=$MODE

accelerate launch --num_processes 8 eval.py --tasks mmlu \
    --num_fewshot 5 \
    --model hnet \
    --model_args checkpoint_path=$CHECKPOINT,config_path=$CONFIG,pretrain_transformer_name=$MODEL_PATH,device=cuda,custom_batch_size=16,max_length=8192,mode=$MODE