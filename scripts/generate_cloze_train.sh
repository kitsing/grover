#!/usr/bin/env bash

### Section1: SBATCH directives to specify job configuration
## job name
#SBATCH --job-name=generate_cloze
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/grover/generate_cloze-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/jobs/grover/generate_cloze-%j.err

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --ntasks-per-node=10
#SBATCH --ntasks=100
#SBATCH --time=2880
#SBATCH --signal=USR1@180 #Signal is sent to batch script itself
## number of tasks per node
## notify 10 minutes before job is killed so we can requeue

CLOZE_PATH=/checkpoint/kitsing/grover/cloze-train
ON_CLUSTER=1

mkdir -p "${CLOZE_PATH}"
export PYTHONPATH=$(pwd)
RUN_STRING="bash cloze/generate_cloze_helper.sh \
            ${CLOZE_PATH}/ \
            42 train"

echo "${RUN_STRING}"
if [[ ${ON_CLUSTER} -eq 1 ]]; then
  srun --label ${RUN_STRING};
else
  eval ${RUN_STRING};
fi
