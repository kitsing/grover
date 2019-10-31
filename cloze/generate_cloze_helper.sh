#!/usr/bin/env bash
FORMAT_NUM=$(printf "%04d" ${SLURM_PROCID})
FNAME="/checkpoint/kitsing/grover/tfrecords/preprocessed_val${FORMAT_NUM}.tfrecord"
OUTPUT_PATH=${1}
CHOICE=${2}
PYTHONPATH=$(pwd) python cloze/generate_cloze_dataset.py --file ${FNAME} --choice ${CHOICE} --output-path ${OUTPUT_PATH}