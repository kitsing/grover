#!/usr/bin/env bash
UNCOND_OUTPUT_DIR="/checkpoint/kitsing/grover/unconditional_samples_dev/${SLURM_PROCID}"
mkdir -p ${UNCOND_OUTPUT_DIR}
PYTHONPATH=$(pwd) python sample/unconditional_generate.py -model_config_fn lm/configs/base.json -model_ckpt /checkpoint/kitsing/grover-models/base/model.ckpt -batch_size 1000000 -dir ${UNCOND_OUTPUT_DIR} -max_batch_size 64 --seed 42