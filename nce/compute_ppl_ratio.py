#!/usr/bin/env python3
from glob import glob
import numpy as np
import tensorflow as tf
from nce.estimate_z import get_g_under_model, compute_z, compute_confidence
from sample.encoder import get_encoder
from scipy.special import logsumexp

def get_tokens(token_file):
    with np.load(token_file) as f:
        return f['tokens']


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', default='.')
    parser.add_argument('--model-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--gen-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--seq-length', default=1025, type=int)
    parser.add_argument('--num-gpus', default=8, type=int)
    parser.add_argument('--dis-ckpt', default='/checkpoint/kitsing/grover-models/base/model.ckpt', type=str)
    parser.add_argument('--gen-ckpt', default='/checkpoint/kitsing/grover-models/base/model.ckpt', type=str)
    parser.add_argument('--noise-files', default='./*.npz', type=str)
    parser.add_argument('--chunk-size', default=512, type=int)
    parser.add_argument('--confidence', default=0.95, type=float)
    parser.add_argument('--sentence-level', action='store_true')
    parser.add_argument('--dis-is-gen2', action='store_true')
    args = parser.parse_args()
    noise_files = glob(args.noise_files)
    encoder = get_encoder()
    inp_tokens = get_tokens(args.inp)
    word_count = np.sum(inp_tokens[:, 1:] != encoder.padding)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    _, log_zs = compute_z(args.batch_size, args.dis_ckpt, args.gen_ckpt, args.gen_config, args.model_config,
                          noise_files, args.num_gpus, args.seq_length, args.chunk_size,
                          dis_is_gen2=args.dis_is_gen2)
    log_z_lower, log_z_upper = compute_confidence(log_zs, args.confidence)

    with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
        compute_prob = get_g_under_model(model_config=args.model_config,
                                         batch_size_per_chunk=args.batch_size,
                                         num_gpus=args.num_gpus,
                                         seq_length=args.seq_length,
                                         dis_ckpt=args.dis_ckpt,
                                         sess=sess, gen_ckpt=args.gen_ckpt,
                                         gen_config=args.gen_config,
                                         dis_is_gen2=args.dis_is_gen2)
        inp_probs_under_model, = tuple(compute_prob(inp_tokens))
    val_score_reshaped = np.reshape(inp_probs_under_model, (-1, 1))
    log_z_reshaped_lower = np.ones_like(val_score_reshaped) * (np.log(args.chunk_size) + log_z_lower)
    log_z_reshaped_upper = np.ones_like(val_score_reshaped) * (np.log(args.chunk_size) + log_z_upper)
    denom_lower = logsumexp(np.concatenate((val_score_reshaped, log_z_reshaped_lower), axis=1), axis=1)
    denom_upper = logsumexp(np.concatenate((val_score_reshaped, log_z_reshaped_upper), axis=1), axis=1)
    nce_lower = np.mean(inp_probs_under_model - denom_lower)
    nce_upper = np.mean(inp_probs_under_model - denom_upper)
    geo_mean_r = np.mean(inp_probs_under_model)

    ppl_reduction = lambda log_z, ratio: ratio * (log_z - geo_mean_r)
    lower_ppl = np.exp(ppl_reduction(log_z_lower, (inp_tokens.shape[0] / word_count)))
    upper_ppl = np.exp(ppl_reduction(log_z_upper, (inp_tokens.shape[0] / word_count)))

    lower_sld = ppl_reduction(log_z_lower, 1.)
    upper_sld = ppl_reduction(log_z_upper, 1.)

    def ci_string(a, b):
        m = (a + b) / 2
        return f'{m} \\pm {abs(m-a)}'
    print(f'nce baseline: {np.log(args.chunk_size + 1)}')
    print(f'nce: {ci_string(nce_lower, nce_upper)}\tppl_reduction: {ci_string(lower_ppl, upper_ppl)}\tsld:{ci_string(lower_sld, upper_sld)}')


if __name__ == '__main__':
    main()
