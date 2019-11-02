#!/usr/bin/env bash
FORMAT_NUM=$(printf "%04d" ${SLURM_PROCID})
OUTPUT_PATH=${1}
CHOICE=${2}
SPLIT=${3}
FNAME="/checkpoint/kitsing/grover/tfrecords/preprocessed_${SPLIT}${FORMAT_NUM}.tfrecord"
PYTHONPATH=$(pwd) python cloze/generate_cloze_dataset.py --file ${FNAME} --choice ${CHOICE} --output-path ${OUTPUT_PATH}