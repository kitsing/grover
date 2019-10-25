#!/usr/bin/env bash

export PYTHONPATH=$(pwd)

learning_rate=1e-4
init_checkpoint=""
max_seq_length=1024
save_checkpoint_steps=1000
K=5

# You can customize the training here
# mega, medium, or base
model_type="base"
OUTPUT_DIR="/checkpoint/kitsing/grover-models/discriminator" # put your output directory here
input_file="/checkpoint/kitsing/grover/tfrecords/*.tfrecord" # put your input files here, it can also be something like "*.tfrecord"
noise_file="/checkpoint/kitsing/grover/unconditional_samples/*.npz" # put your input files here, it can also be something like "*.tfrecord"

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
num_train_steps=800000

# Make sure batch size scales.
let batch_size=1

NODE_LIST=$( scontrol show hostname ${SLURM_JOB_NODELIST} | sed -z 's/\n/\:8,/g' )
NODE_LIST=${NODE_LIST%?}

mpirun --allow-run-as-root --tag-output -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib -x CONDA_SHLVL -x LD_LIBRARY_PATH -x LS_COLORS -x CONDA_EXE -x SRUN_DEBUG -x SLURM_STEP_ID -x SLURM_STEP_GPUS -x SLURM_NODEID -x SLURM_TASK_PID -x SSH_CONNECTION -x SLURM_PRIO_PROCESS -x SLURM_CPU_BIND_VERBOSE -x CUDNN_LIB_DIR -x LANG -x SLURM_SUBMIT_DIR -x LESS -x HOSTNAME -x SLURM_CPUS_PER_TASK -x SLURM_STEPID -x SLURM_SRUN_COMM_HOST -x SLURM_DISTRIBUTION -x ENVIRONMENT -x PATH_modshare -x LOADEDMODULES_modshare -x FPATH -x CONDA_PREFIX -x JAVA_HOME -x SLURM_PROCID -x SLURM_JOB_GID -x DYLD_LIBRARY_PATH -x SLURM_CPU_BIND -x SLURMD_NODENAME -x ZSH -x SLURM_TASKS_PER_NODE -x S_COLORS -x LD_LIBRARY_PATH_modshare -x CC -x XDG_SESSION_ID -x MODULES_CMD -x LIBRARY_PATH_modshare -x USER -x SLURM_NNODES -x ENV -x SLURM_LAUNCH_NODE_IPADDR -x PAGER -x LSCOLORS -x SLURM_STEP_TASKS_PER_NODE -x PWD -x SLURM_JOB_NODELIST -x HOME -x SLURM_CLUSTER_NAME -x CONDA_PYTHON_EXE -x LC_TERMINAL -x CMAKE_PREFIX_PATH -x SLURM_NODELIST -x SSH_CLIENT -x MODULES_MODSHARE_DYLD_LIBRARY_PATH -x CUDA_HOME -x CPATH -x SLURM_NTASKS -x TMUX -x SLURM_UMASK -x LC_TERMINAL_VERSION -x SLURM_JOB_CPUS_PER_NODE -x BASH_ENV -x XDG_DATA_DIRS -x SLURM_TOPOLOGY_ADDR -x _LMFILES__modshare -x SLURM_WORKING_CLUSTER -x SLURM_STEP_NODELIST -x SLURM_JOB_NAME -x SLURM_SRUN_COMM_PORT -x TMPDIR -x LIBRARY_PATH -x SLURM_JOB_GPUS -x SLURM_JOBID -x LOADEDMODULES -x SLURM_JOB_QOS -x SLURM_TOPOLOGY_ADDR_PATTERN -x CONDA_PROMPT_MODIFIER -x SLURM_LABELIO -x MAIL -x SLURM_CPUS_ON_NODE -x CXX -x MPI_HOME -x SLURM_JOB_NUM_NODES -x SLURM_MEM_PER_NODE -x SHELL -x TERM -x CMAKE_PREFIX_PATH_modshare -x SLURM_JOB_UID -x MANPATH_modshare -x SLURM_JOB_PARTITION -x SLURM_CPU_BIND_LIST -x CUDNN_INCLUDE_DIR -x SLURM_JOB_USER -x CUDA_VISIBLE_DEVICES -x TMUX_PANE -x SLURM_NPROCS -x SHLVL -x LANGUAGE -x SLURM_SUBMIT_HOST -x PYTHONPATH -x SLURM_JOB_ACCOUNT -x MANPATH -x SLURM_STEP_LAUNCHER_PORT -x MODULEPATH -x SLURM_GTIDS -x LOGNAME -x DBUS_SESSION_BUS_ADDRESS -x XDG_RUNTIME_DIR -x MODULEPATH_modshare -x PATH -x SLURM_JOB_ID -x _LMFILES_ -x SLURM_CPU_BIND_TYPE -x SLURM_STEP_NUM_TASKS -x MODULESHOME -x CONDA_DEFAULT_ENV -x NCCL_ROOT_DIR -x CUDNN_ROOT_DIR -x ET_VERSION -x SLURM_STEP_NUM_NODES -x SLURM_LOCALID -x GPU_DEVICE_ORDINAL -x HOROVOD_STALL_CHECK_TIME_SECONDS -x HOROVOD_STALL_SHUTDOWN_TIME_SECONDS -x HOROVOD_NUM_NCCL_STREAMS -x HOROVOD_MLSL_BGT_AFFINITY -x HOROVOD_GLOO_TIMEOUT_SECONDS --map-by ppr:4:socket -mca plm_rsh_agent "ssh -q -o StrictHostKeyChecking=no" -mca btl_tcp_if_exclude lo,docker0 --oversubscribe  -np ${SLURM_NTASKS} -H ${NODE_LIST} python lm/train_nce.py \
    --config_file=lm/configs/${model_type}.json \
    --input_file=${input_file} \
    --noise_file=${noise_file} \
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
    --k=${K}
