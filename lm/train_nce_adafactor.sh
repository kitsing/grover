#!/usr/bin/env bash
# module purge
# module load anaconda3/5.0.1 cuda/10.0 git/2.15.1/gcc.5.4.0 gcc/7.1.0 tmux/2.8/gcc.7.3.0 vim/8.1.0788/gcc.7.3.0 torch/080919/cudnn.7.6.2 NCCL/2.4.8-1-cuda.10.0 openmpi/3.1.1/gcc.7.3.0 java
# . /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
# conda deactivate
# conda activate grover
export PYTHONPATH=$(pwd)

learning_rate=1e-4
max_seq_length=1024
save_checkpoint_steps=1000
K=1

# You can customize the training here
# mega, medium, or base
OUTPUT_DIR=${1} # put your output directory here
init_checkpoint=${2}
model_type=${3}
gen_checkpoint=${4}
input_file="/checkpoint/kitsing/grover/tfrecords/preprocessed_train*.tfrecord" # put your input files here, it can also be something like "*.tfrecord"
input_dev_file="/checkpoint/kitsing/grover/tfrecords/preprocessed_valid0[0-5]*.tfrecord" # put your input files here, it can also be something like "*.tfrecord"

noise_file="/checkpoint/kitsing/grover/unconditional_samples_vanilla/[0-6]/*.npz"
noise_dev_file="/checkpoint/kitsing/grover/unconditional_samples_vanilla/7/*.npz"
# old noise files below
#noise_file="/checkpoint/kitsing/grover/unconditional_samples_dev/[0-6]/*.npz"
#noise_dev_file="/checkpoint/kitsing/grover/unconditional_samples_dev/7/*.npz"

num_tpu_cores=32
batch_size_per_core=16

if [[ ${init_checkpoint} == "0" ]]; then
  init_checkpoint=""
fi

if [[ ${gen_checkpoint} == "0" ]]; then
  gen_checkpoint=""
fi

if [[ ${model_type} == "base" ]]; then
    num_tpu_cores=32
    batch_size_per_core=16
elif [[ ${model_type} == "medium" ]]; then
    num_tpu_cores=128
    batch_size_per_core=4
elif [[ ${model_type} == "mega" ]]; then
    num_tpu_cores=256
    batch_size_per_core=2
fi


# there are 20k * 1024 examples so this translates to 20 epochs. seems ok and i can run for more if needed
num_train_steps=800000

# Make sure batch size scales.
let batch_size=6

# NODE_LIST=$( scontrol show hostname ${SLURM_JOB_NODELIST} | sed -z 's/\n/\:8,/g' )
# NODE_LIST=${NODE_LIST%?}

# export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"

python lm/train_nce.py \
    --config_file=lm/configs/${model_type}.json \
    --input_file=${input_file} \
    --input_dev_file=${input_dev_file} \
    --noise_file=${noise_file} \
    --noise_dev_file=${noise_dev_file} \
    --output_dir=${OUTPUT_DIR} \
    --max_seq_length=${max_seq_length} \
    --train_batch_size=${batch_size} \
    --learning_rate=${learning_rate} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=10000 \
    --save_checkpoints_steps=${save_checkpoint_steps} \
    --iterations_per_loop=${save_checkpoint_steps} \
    --use_tpu=False \
    --num_tpu_cores=$num_tpu_cores \
    --init_checkpoint=${init_checkpoint} \
    --gen_checkpoint=${gen_checkpoint} \
    --k=${K}
