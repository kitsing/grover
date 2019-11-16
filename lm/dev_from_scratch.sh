#!/usr/bin/env bash

export PYTHONPATH=$(pwd)

learning_rate=1e-4
max_seq_length=1024
save_checkpoint_steps=100

# You can customize the training here
# mega, medium, or base
model_type="base"
OUTPUT_DIR=${1} # put your output directory here
init_checkpoint=${2}

input_file="/checkpoint/kitsing/grover/tfrecords/preprocessed_val0[1-8]*.tfrecord" # put your input files here, it can also be something like "*.tfrecord"

if [ ${model_type} == "base" ]; then
    num_tpu_cores=32
    batch_size_per_core=16
elif [ ${model_type} == "medium" ]; then
    num_tpu_cores=128
    batch_size_per_core=4
elif [ ${model_type} == "mega" ]; then
    num_tpu_cores=256
    batch_size_per_core=2
fi


# there are 20k * 1024 examples so this translates to 20 epochs. seems ok and i can run for more if needed
num_train_steps=852000

# Make sure batch size scales.
let batch_size="8"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"

python lm/finetune_mirrored.py \
    --config_file=lm/configs/${model_type}.json \
    --input_file=${input_file} \
    --output_dir=${OUTPUT_DIR} \
    --max_seq_length=${max_seq_length} \
    --train_batch_size=${batch_size} \
    --learning_rate=${learning_rate} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=10000 \
    --save_checkpoints_steps=${save_checkpoint_steps} \
    --iterations_per_loop=${save_checkpoint_steps} \
    --use_tpu=False \
    --tpu_name=$(hostname) \
    --num_tpu_cores=$num_tpu_cores \
    --init_checkpoint=${init_checkpoint}
