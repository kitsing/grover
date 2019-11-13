#!/usr/bin/env bash
NITER=${1}
model=${2}
ID=${3}
OUTPUT_DIR=/checkpoint/kitsing/grover/${ID}/nce-${NITER}
mkdir -p ${OUTPUT_DIR}

PYTHONPATH="." python nce/calculate_nce_loss.py --model-config-fn lm/configs/${model}.json --noise-model-config-fn lm/configs/base.json --gen-model-ckpt /checkpoint/kitsing/grover-models/base/model.ckpt --dis-model-ckpt /checkpoint/kitsing/grover-models/discriminator-${ID}/model.ckpt-${NITER} --noise-model-ckpt /checkpoint/kitsing/grover-models/base/model.ckpt --batch-size 8 --num-noise-chunks 1 --file /checkpoint/kitsing/grover/nce/0.npz --output-path ${OUTPUT_DIR} --num-gpus 8  --noise-output-path /checkpoint/kitsing/grover/nce/noises/${NITER}
