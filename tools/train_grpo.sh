#!/bin/bash
NCCL_LOG="YOUR_PATH/ncll_all.log"
MAIN_LOG="YOUR_PATH/grpo_all.log"

mkdir -p "$(dirname "$NCCL_LOG")"
mkdir -p "$(dirname "$MAIN_LOG")"

> "$NCCL_LOG"
> "$MAIN_LOG"

echo "=== GRPO training started at $(date) ===" >> "$MAIN_LOG"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8

export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)

export PYTHONPATH="YOUR_PATH/tools:${PYTHONPATH}"

export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_BUFFSIZE=2097152
export NCCL_NTHREADS=16
export NCCL_TIMEOUT=3000
export NCCL_DEBUG_FILE="$NCCL_LOG"

echo "Environment variable check:" >> "$MAIN_LOG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" >> "$MAIN_LOG"
echo "NPROC_PER_NODE: $NPROC_PER_NODE" >> "$MAIN_LOG"
echo "CC: $CC" >> "$MAIN_LOG"
echo "CXX: $CXX" >> "$MAIN_LOG"
echo "NCCL_DEBUG_FILE: $NCCL_LOG" >> "$MAIN_LOG"

exec > >(tee -a "$MAIN_LOG") 2>&1


swift rlhf \
    --rlhf_type grpo \
    --model YOUR_MODEL_PATH \
    --model_type qwen2_5_vl \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --reward_funcs format geo_format triple_accuracy think_answer geoscore_accuracy  \
    --reward_weights 1.0 1.0 1.0 0.5 1.5 \
    --use_vllm true \
    --vllm_mode colocate \
    --offload_optimizer false \
    --offload_model false \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_lora_rank 128 \
    --move_model_batches 8 \
    --torch_dtype bfloat16 \
    --dataset YOUR_DATASET_PATH \
    --max_completion_length 1024 \
    --max_pixels 200000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 200 \
    --logging_steps 1 \
    --output_dir YOUR_OUTPUT_DIR \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --num_generations 8 \
    --temperature 0.7 \
    --system YOUR_PATH/tools/grpo_prompt_en.txt \
    --deepspeed zero3 \
    --log_completions true \
    --report_to swanlab \
    --swanlab_project YOUR_PROJECT_NAME \
    --swanlab_exp_name  YOUR_EXP_NAME \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --beta 0.001 \
    --num_iterations 1

echo "=== GRPO training finished at $(date) ===" >> "$MAIN_LOG"
