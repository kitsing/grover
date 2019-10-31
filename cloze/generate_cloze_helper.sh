#!/usr/bin/env bash
FNAME=ls /checkpoint/kitsing/grover/tfrecords/preprocessed_val$(printf "%04d" ${SLURM_PROCID}).tfrecord
OUTPUT_PATH=${1}
CHOICE=${2}
PYTHONPATH=$(pwd) python cloze/generate_cloze_dataset.py --file ${FNAME} --choice ${CHOICE} --output-path ${OUTPUT_PATH}