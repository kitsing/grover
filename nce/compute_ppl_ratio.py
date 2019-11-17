#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from nce.estimate_z import get_g_under_model


def get_tokens(token_file):
    with np.load(token_file) as f:
        return f['tokens']


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', default='.')
    parser.add_argument('--model-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--seq-length', default=1025, type=int)
    parser.add_argument('--num-gpus', default=8, type=int)
    parser.add_argument('--dis-ckpt', default='/checkpoint/kitsing/grover-models/base/model.ckpt')
    args = parser.parse_args()
    inp_tokens = get_tokens(args.inp)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
        compute_prob = get_g_under_model(model_config=args.model_config,
                                         batch_size_per_chunk=args.batch_size,
                                         num_gpus=args.num_gpus,
                                         seq_length=args.seq_length,
                                         dis_ckpt=args.dis_ckpt,
                                         sess=sess, )
        inp_probs_under_model, = tuple(compute_prob(inp_tokens))

    print(np.mean(inp_probs_under_model))


if __name__ == '__main__':
    main()
