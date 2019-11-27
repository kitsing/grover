#!/usr/bin/env python3
import numpy as np
from scipy.special import logsumexp
import tensorflow as tf
from nce.eval_nce_against_baseline import get_dirty_noises
from sample.encoder import get_encoder


def get_g_under_model(model_config, batch_size_per_chunk, num_gpus, seq_length,
                      dis_ckpt, sess, gen_config, gen_ckpt):
    from sample.encoder import get_encoder
    from lm.modeling import GroverConfig, eval_seq
    from nce.utils import get_seq_probs
    from nce.utils import restore
    encoder = get_encoder()
    news_config = GroverConfig.from_json_file(model_config)
    gen_news_config = GroverConfig.from_json_file(gen_config)

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
                                           ignore_ids=ignore_ids, discriminator_only=True,
                                           gen_config=gen_news_config))
            all_gs.append(gs)

    with tf.device('/cpu:0'):
        merged_gs = tf.concat(all_gs, axis=0)

    restore('dis', dis_ckpt, sess)
    if news_config.non_residual:
        assert gen_config is not None
        restore('gen', gen_ckpt, sess)

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
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--noises')
    parser.add_argument('--model-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--gen-config', default=None)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--seq-length', default=1025, type=int)
    parser.add_argument('--num-gpus', default=8, type=int)
    parser.add_argument('--dis-ckpt', default='/checkpoint/kitsing/grover-models/base/model.ckpt')
    parser.add_argument('--gen-ckpt', default=None)
    parser.add_argument('--output-path', default=None, type=str)
    parser.add_argument('--confidence', default=0.95, type=float)
    args = parser.parse_args()
    noise_files = glob(args.noises)
    model_config = args.model_config
    batch_size = args.batch_size
    num_gpus = args.num_gpus
    seq_length = args.seq_length
    dis_ckpt = args.dis_ckpt
    gen_ckpt = args.gen_ckpt
    gen_config = args.gen_config
    gs_under_model, log_zs = compute_z(batch_size,
                                       dis_ckpt, gen_ckpt,
                                       gen_config, model_config,
                                       noise_files, num_gpus,
                                       seq_length)

    if args.output_path is not None:
        np.savez(f'{args.output_path}', gs=gs_under_model)

    print(compute_confidence(log_zs, args.confidence))


def compute_confidence(log_zs, confidence: float = 0.95):
    from scipy.stats import sem, t
    samples = np.exp(log_zs)
    m = np.mean(samples)
    std_err = sem(samples)
    h = std_err * t.ppf((1 + confidence) / 2, len(log_zs) - 1)
    return np.log(m - h), np.log(m + h)


def compute_z(batch_size, dis_ckpt,
              gen_ckpt, gen_config,
              model_config, noise_files,
              num_gpus, seq_length, chunk_size: int = 512):
    encoder = get_encoder()
    noise_tokens = get_dirty_noises(noise_files, eoa=encoder.__dict__['end_article'],
                                    pad=encoder.padding, seq_length=seq_length)
    print(f'noise shape: {noise_tokens.shape} num of chunks: {noise_tokens.shape[0] // chunk_size}')
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
        compute_g = get_g_under_model(model_config,
                                      batch_size, num_gpus,
                                      seq_length,
                                      dis_ckpt,
                                      sess, gen_config=gen_config, gen_ckpt=gen_ckpt)
        gs_under_model = compute_g(noise_tokens)[0]
    log_zs = []
    for chunk in range(noise_tokens.shape[0] // chunk_size):
        gs_under_model_chunk = gs_under_model[chunk*chunk_size:(chunk+1)*chunk_size]
        log_z = logsumexp(gs_under_model_chunk) - np.log(float(gs_under_model_chunk.shape[0]))
        log_zs.append(log_z)
    return gs_under_model, log_zs


if __name__ == '__main__':
    main()
