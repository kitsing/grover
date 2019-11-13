#!/usr/bin/env bash
### Section1: SBATCH directives to specify job configuration
## job name
#SBATCH --job-name=train-grover-nce-dev-corrected
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/grover/train-grover-nce-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/jobs/grover/train-grover-nce-%j.err
#SBATCH --time=2880

## partition name
#SBATCH --partition=priority
#SBATCH --comment="acl 2020 submission: need this quick run to see if the architecture makes sense"
## number of nodes
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80   # 80/8 cpus per task
## number of tasks per node
#SBATCH --gres=gpu:8
#SBATCH --constraint=volta32gb
srun --label lm/train_nce_adafactor_dev_corrected.sh ${1} ${2} ${3} ${4}
