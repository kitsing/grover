#!/usr/bin/env bash

### Section1: SBATCH directives to specify job configuration
## job name
#SBATCH --job-name=preprocess
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/grover/preprocess-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/jobs/grover/preprocess-%j.err

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=32
#SBATCH --ntasks=1024
#SBATCH --time=2880
#SBATCH --signal=USR1@180 #Signal is sent to batch script itself
## number of tasks per node
## notify 10 minutes before job is killed so we can requeue

MODEL_PATH=/checkpoint/kitsing/grover/tfrecords
DATA_PATH=/checkpoint/kitsing/realnews_corpus/realnews/realnews.jsonl
NUM_FOLDS=${SLURM_NTASKS}
ON_CLUSTER=1

mkdir -p "${MODEL_PATH}"
export PYTHONPATH=$(pwd)
RUN_STRING="python realnews/prepare_unconditioned_lm_data.py \
            -num_folds ${NUM_FOLDS}
            -base_fn ${MODEL_PATH}/preprocessed_ \
            -input_fn ${DATA_PATH}"

echo "${RUN_STRING}"
if [[ ${ON_CLUSTER} -eq 1 ]]; then
  srun --label ${RUN_STRING};
else
  eval ${RUN_STRING};
fi
