#!/bin/bash

DATASET_1="/mnt/cache/shanhang/newhome/project/fork/Megatron-LLaMA/pdopt/70B80layers_output_document"
DATASET_2="/mnt/cache/shanhang/newhome/project/fork/Megatron-LLaMA/pdopt/70B80layers_output_document"
DATASET_3="/mnt/cache/shanhang/newhome/project/fork/Megatron-LLaMA/pdopt/70B80layers_output_document"
DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"

TP_SIZE=8
PP_SIZE=4
WORLD_SIZE=32
MICRO_BATCH_SIZE=1
# The int is the number of micro steps of gradient accumulation
GLOBAL_BATCH_SIZE=$((($WORLD_SIZE * $MICRO_BATCH_SIZE) / ($TP_SIZE * $PP_SIZE) * 8))

JOB_NAME="LLaMA_tp${TP_SIZE}_pp${PP_SIZE}_mbs${MICRO_BATCH_SIZE}_gpus${WORLD_SIZE}"

# SAVE_CHECKPOINT_PATH="/mnt/lustrenew/shanhang/project/fork/Megatron-LLaMA/save_ckpt"
TOKENIZER_PATH="/mnt/lustrenew/share_data/PAT/datasets/llama2/70B/"
TENSORBOARD_DIR="/mnt/lustrenew/shanhang/project/fork/Megatron-LLaMA/tensorboard"

TRAIN_ITERS=1000
EVAL_ITERS=10
EVAL_INTERVAL=1000
SAVE_INTERVAL=100
LOG_INTERVAL=1

# Setting --tensorboard-queue-size to 1 significantly slows down the training
options=" \
    --sequence-parallel \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers 80 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 2048 \
        --max-position-embeddings 4096 \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --swiglu \
        --ffn-hidden-size 11008\
        --disable-bias-linear \
        --RMSNorm \
        --layernorm-epsilon 1e-6 \
        --causal-lm \
    --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --make-vocab-size-divisible-by 1 \
    --init-method-std 0.01 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr 6.0e-5 \
        --lr-decay-iters 10 \
        --lr-warmup-iters 5 \
        --min-lr 6.0e-6 \
        --override-opt_param-scheduler \
        --lr-decay-style cosine \
    --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --overlapped-distributed-optimizer \
        --reduce-bucket-size=2e8 \
        --no-gradient-accumulation-fusion \
    --dataloader-type cyclic \
        --data-impl mmap \
        --data-path ${DATASET} \
        --split 98,2,0 \
    --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
    --log-interval ${LOG_INTERVAL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
        --tensorboard-queue-size 1000 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
    --job-name ${JOB_NAME} \
    --fp16 \
    --recompute-activations \
        --recompute-granularity selective \
    --use-flash-attn
    "

export CUDA_DEVICE_MAX_CONNECTIONS=1
set -euxo pipefail
IFS="," read -ra gpus_arr <<< "$SLURM_STEP_GPUS"
gpus_per_node=${#gpus_arr[@]}
num_processes=$((SLURM_NNODES * gpus_per_node))
head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

GPUS_PER_NODE="$gpus_per_node"
NNODES="$SLURM_NNODES"
NODE_RANK="$SLURM_PROCID"
MASTER_ADDR="$head_node_ip"
MASTER_PORT=29512
WORLD_SIZE=$((GPUS_PER_NODE*NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
    ${options} \
