#!/usr/bin/env python3
import numpy as np
from scipy.special import logsumexp
import tensorflow as tf
from nce.eval_nce_against_baseline import get_dirty_noises
from sample.encoder import get_encoder

def get_g_under_model(model_config, batch_size_per_chunk, num_gpus, seq_length,
                      dis_ckpt, sess):
    from sample.encoder import get_encoder
    from lm.modeling import GroverConfig, eval_seq
    from nce.utils import get_seq_probs
    from nce.utils import restore
    encoder = get_encoder()
    news_config = GroverConfig.from_json_file(model_config)

    all_tokens = []
    all_gs = []

    ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])
    ignore_ids_np = np.array(encoder.special_tokens_onehot)
    ignore_ids_np[encoder.__dict__['end_article']] = 0
    for i in range(num_gpus):
        with tf.device('/gpu:' + str(i)):
            # actual examples
            tokens = tf.placeholder(tf.int32, [batch_size_per_chunk, seq_length])
            all_tokens.append(tokens)
            gs = tf.stop_gradient(eval_seq(news_config, tokens, 1., baseline=False,
                                           ignore_ids=ignore_ids, discriminator_only=True))
            all_gs.append(gs)

    with tf.device('/cpu:0'):
        merged_gs = tf.concat(all_gs, axis=0)

    restore('dis', dis_ckpt, sess)

    gs_under_model = lambda inp: get_seq_probs(seqs=inp,
                                               batch_size=batch_size_per_chunk * num_gpus,
                                               token_place_holders=all_tokens,
                                               num_gpus=num_gpus,
                                               tf_outputs=[merged_gs],
                                               ignore_ids_np=ignore_ids_np,
                                               ignore_ids=ignore_ids, sess=sess, seq_length=seq_length)
    return gs_under_model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--noises')
    parser.add_argument('--model-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--noise-model-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--seq-length', default=1025, type=int)
    parser.add_argument('--num-gpus', default=8, type=int)
    parser.add_argument('--dis-ckpt', default='/checkpoint/kitsing/grover-models/base/model.ckpt')
    parser.add_argument('--output-path', default=None, type=str)
    args = parser.parse_args()
    from glob import glob
    encoder = get_encoder()
    noise_files = glob(args.noises)
    noise_tokens = get_dirty_noises(noise_files, eoa=encoder.__dict__['end_article'],
                                    pad=encoder.padding, seq_length=args.seq_length)
    print(f'noise shape: {noise_tokens.shape}')
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
        compute_g = get_g_under_model(args.model_config,
                                      args.batch_size, args.num_gpus,
                                      args.seq_length,
                                      args.dis_ckpt,
                                      sess)
        gs_under_model = compute_g(noise_tokens)[0]
    if args.output_path is not None:
        np.savez(f'{args.output_path}', gs=gs_under_model)
    print(np.exp(logsumexp(gs_under_model) - np.log(float(gs_under_model.shape[0]))))


if __name__ == '__main__':
    main()
