#!/usr/bin/env bash
### Section1: SBATCH directives to specify job configuration
## job name
#SBATCH --job-name=eval-cloze
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/grover/eval-cloze-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/jobs/grover/eval-cloze-%j.err
#SBATCH --time=2880

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80   # 80/8 cpus per task
## number of tasks per node
#SBATCH --gres=gpu:8
#SBATCH --constraint=volta32gb
export PYTHONPATH=$(pwd)

OUTPUT_DIR=${1} # put your output directory here
mkdir -p ${OUTPUT_DIR}
DIS_MODEL_CKPT=${2}
model_type=${3}
#input_dev_file='/checkpoint/kitsing/grover/cloze/preprocessed_val0[0-5]*.tfrecord.npz'
input_dev_file='/checkpoint/kitsing/grover/cloze/preprocessed_val00[0-9]*.tfrecord.npz'

# Make sure batch size scales.
let batch_size=160

# NODE_LIST=$( scontrol show hostname ${SLURM_JOB_NODELIST} | sed -z 's/\n/\:8,/g' )
# NODE_LIST=${NODE_LIST%?}

# export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"

set -o noglob
RUN_STRING="python lm/eval_nce.py \
--model-config-fn lm/configs/${model_type}.json \
--gen-model-ckpt /checkpoint/kitsing/grover-models/${model_type}/model.ckpt \
--dis-model-ckpt ${DIS_MODEL_CKPT} \
--batch-size ${batch_size} \
--files ${input_dev_file} \
--output-path ${OUTPUT_DIR} \
--num-gpus 8"
srun --label ${RUN_STRING}
set +o noglob
