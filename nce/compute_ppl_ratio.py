#!/usr/bin/env python3
from glob import glob
import numpy as np
import tensorflow as tf
from nce.estimate_z import get_g_under_model, compute_z, compute_confidence
from sample.encoder import get_encoder

def get_tokens(token_file):
    with np.load(token_file) as f:
        return f['tokens']


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', default='.')
    parser.add_argument('--model-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--gen-config', default=None)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--seq-length', default=1025, type=int)
    parser.add_argument('--num-gpus', default=8, type=int)
    parser.add_argument('--dis-ckpt', default='/checkpoint/kitsing/grover-models/base/model.ckpt', type=str)
    parser.add_argument('--gen-ckpt', default=None, type=str)
    parser.add_argument('--noise-files', default='./*.npz', type=str)
    parser.add_argument('--chunk-size', default=512, type=int)
    parser.add_argument('--confidence', default=0.95, type=float)
    parser.add_argument('--sentence-level', action='store_true')
    args = parser.parse_args()
    noise_files = glob(args.noise_files)
    encoder = get_encoder()
    inp_tokens = get_tokens(args.inp)
    word_count = np.sum(inp_tokens[:, 1:] != encoder.padding)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    _, log_zs = compute_z(args.batch_size, args.dis_ckpt, args.gen_ckpt, args.gen_config, args.model_config,
                          noise_files, args.num_gpus, args.seq_length, args.chunk_size)
    lower_log_z, upper_log_z = compute_confidence(log_zs, args.confidence)
    with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
        compute_prob = get_g_under_model(model_config=args.model_config,
                                         batch_size_per_chunk=args.batch_size,
                                         num_gpus=args.num_gpus,
                                         seq_length=args.seq_length,
                                         dis_ckpt=args.dis_ckpt,
                                         sess=sess, gen_ckpt=args.gen_ckpt,
                                         gen_config=args.gen_config)
        inp_probs_under_model, = tuple(compute_prob(inp_tokens))

    geo_mean_r = np.mean(inp_probs_under_model)
    if args.sentence_level:
        s_w_ratio = 1.
    else:
        s_w_ratio = (inp_tokens.shape[0] / word_count)
    ppl_reduction = lambda log_z: np.exp( s_w_ratio * (log_z - geo_mean_r) )
    lower_ppl = ppl_reduction(lower_log_z)
    upper_ppl = ppl_reduction(upper_log_z)

    def ci_string(a, b):
        m = (a + b) / 2
        return f'{m} \\pm {abs(m-a)}'
    print(f'ppl_reduction: {ci_string(lower_ppl, upper_ppl)}')


if __name__ == '__main__':
    main()
