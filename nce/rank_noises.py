#!/usr/bin/env python3
import numpy as np
from scipy.special import logsumexp
import tensorflow as tf
from nce.eval_nce_against_baseline import get_dirty_noises
from sample.encoder import get_encoder


def main():
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--noises')
    parser.add_argument('--model-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--gen-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--seq-length', default=1025, type=int)
    parser.add_argument('--num-gpus', default=8, type=int)
    parser.add_argument('--dis-ckpt', default='/checkpoint/kitsing/grover-models/base/model.ckpt')
    parser.add_argument('--gen-ckpt', default='/checkpoint/kitsing/grover-models/base/model.ckpt')
    parser.add_argument('--output-path', default=None, type=str)
    args = parser.parse_args()
    noise_files = glob(args.noises)
    model_config = args.model_config
    batch_size = args.batch_size
    num_gpus = args.num_gpus
    seq_length = args.seq_length
    dis_ckpt = args.dis_ckpt
    gen_ckpt = args.gen_ckpt
    gen_config = args.gen_config
    gs_under_model, texts = get_ranked_noises(batch_size,
                                              dis_ckpt, gen_ckpt,
                                              gen_config, model_config,
                                              noise_files, num_gpus,
                                              seq_length)
    from csv import DictWriter
    with open(args.output_path, mode='w', newline='') as f:
        writer = DictWriter(f, fieldnames=['score', 'text'])
        writer.writeheader()
        for score, text in zip(gs_under_model, texts):
            writer.writerow({'score': score, 'text': text})


def get_ranked_noises(batch_size, dis_ckpt,
                      gen_ckpt, gen_config,
                      model_config, noise_files,
                      num_gpus, seq_length,
                      dis_is_gen2: bool = False):
    from nce.estimate_z import get_g_under_model
    encoder = get_encoder()
    noise_tokens = get_dirty_noises(noise_files, eoa=encoder.__dict__['end_article'],
                                    pad=encoder.padding, seq_length=seq_length)
    print(f'noise shape: {noise_tokens.shape}')
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
        compute_g = get_g_under_model(model_config,
                                      batch_size, num_gpus,
                                      seq_length,
                                      dis_ckpt,
                                      sess, gen_config=gen_config, gen_ckpt=gen_ckpt,
                                      dis_is_gen2=dis_is_gen2)
        gs_under_model = compute_g(noise_tokens)[0]
    texts = []
    for n in noise_tokens:
        texts.append(encoder.decode([_ for _ in n if _ != encoder.padding]))
    return gs_under_model, texts


if __name__ == '__main__':
    main()
