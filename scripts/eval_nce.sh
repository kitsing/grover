#!/usr/bin/env bash
NITER=${1}
mkdir -p /checkpoint/kitsing/grover/nce-${NITER}-coupled
rm -rf /checkpoint/kitsing/grover/nce-${NITER}-coupled/*

PYTHONPATH="." python nce/calculate_nce_loss.py --model-config-fn lm/configs/base_mask_padding_sum_coupled.json --noise-model-config-fn lm/configs/base.json --gen-model-ckpt /checkpoint/kitsing/grover-models/base/model.ckpt --dis-model-ckpt /checkpoint/kitsing/grover-models/discriminator-debugging-coupled/model.ckpt-${NITER} --noise-model-ckpt /checkpoint/kitsing/grover-models/base/model.ckpt --batch-size 8 --num-noise-chunks 1 --file /checkpoint/kitsing/grover/nce/0.npz --output-path /checkpoint/kitsing/grover/nce-${NITER}-coupled --num-gpus 8  --noise-output-path /checkpoint/kitsing/grover/nce/noises/${NITER}
