#!/bin/bash

# 设置默认值，如果环境变量未定义
WORLD_SIZE=${WORLD_SIZE:-2}
RANK=${RANK:-0}

# 设置环境变量
export NNODES=1
export NODE_RANK=$RANK
export NPROC_PER_NODE=2
export CUDA_VISIBLE_DEVICES=0,1



swift sft \
    --model YOUR_MODEL_PATH \
    --model_type qwen2_5_vl \
    --train_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --target_modules all-linear \
    --dataset YOUR_DATASET_PATH \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --max_pixels 285043 \
    --save_strategy epoch \
    --logging_steps 5 \
    --output_dir YOUR_OUTPUT_DIR \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --report_to swanlab \
    --deepspeed zero3 \
    --gradient_checkpointing true


