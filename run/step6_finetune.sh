#!/bin/bash

source base_training_args.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
train_files=$1
model_path=$2
job_name=$3
devices=$4



export NUM_PROC=$(echo $devices | wc -w)


# export CUDA_VISIBLE_DEVICES=$devices

output_dir=../out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

# use fsdp for large models
if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
    elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
    elif [[ $model_path == "meta-llama/Llama-2-7b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_7b_finetune"
    elif [[ $model_path == "meta-llama/Meta-Llama-3.1-8B" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_7b_finetune"
    elif [[ $model_path == "meta-llama/Meta-Llama-3.1-8B-Instruct" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_7b_finetune"
fi

training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"


header="torchrun --nproc_per_node $NUM_PROC --nnodes 1 \
--rdzv-id=$RANDOM --rdzv_backend c10d --rdzv-endpoint=localhost:9996 \
../rose/model_training.py"

echo "$header $training_args"
eval "$header" "$training_args"

