### Section1: SBATCH directives to specify job configuration
## job name
#SBATCH --job-name=train-grover-nce
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/grover/train-grover-nce-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/jobs/grover/preproctrain-grover-nceess-%j.err
#SBATCH --time=2880

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=2

## number of tasks per node
#SBATCH --gres=gpu:8
#SBATCH --constraint=volta32gb

export OMPI_MCA_btl_openib_allow_ib=1

mpirun -np 16 python $HOME/pythonfile.py --batch_size=128 --epochs=90