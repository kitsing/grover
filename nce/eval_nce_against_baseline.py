#!/usr/bin/env python3
import numpy as np
from scipy.special import logsumexp
import tensorflow as tf


def compute_nce_probs(inp, noise_probs, noise_probs_under_model):
    loaded = np.load(inp)
    noise_weighted = noise_probs_under_model - noise_probs
    true_weighted = loaded['input_probs_under_model'] - loaded['input_probs_under_noise']
    z = np.concatenate(
        (true_weighted.reshape((-1, 1)), np.tile(noise_weighted.reshape((1, -1)), (true_weighted.shape[0], 1))), axis=1)
    probs = true_weighted - logsumexp(z, axis=1)
    return probs


def get_all_noises(noise_files):
    all_noises = []
    all_noise_probs = []
    for f in noise_files:
        loaded_noise = np.load(f)
        all_noises.append(loaded_noise['noise_tokens'])
        all_noise_probs.append(loaded_noise['noise_probs_under_noise'])
    return np.concatenate(all_noises, axis=0), np.concatenate(all_noise_probs, axis=0)


def compute_prob_under_model(inp, noise_model_config, model_config, batch_size_per_chunk, num_gpus, seq_length,
                             gen_ckpt, dis_ckpt):
    from sample.encoder import get_encoder
    from lm.modeling import GroverConfig, sample
    from nce.calculate_nce_loss import eval_seq, restore, get_seq_probs
    encoder = get_encoder()
    news_config = GroverConfig.from_json_file(model_config)

    tf_config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
        all_tokens = []
        all_probs = []

        ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])
        ignore_ids_np = np.array(encoder.special_tokens_onehot)
        ignore_ids_np[encoder.__dict__['end_article']] = 0
        for i in range(num_gpus):
            with tf.device('/gpu:' + str(i)):
                # actual examples
                tokens = tf.placeholder(tf.int32, [batch_size_per_chunk, seq_length])
                all_tokens.append(tokens)
                probs = tf.stop_gradient(eval_seq(news_config, tokens, 1., baseline=False,
                                                  ignore_ids=ignore_ids, gen_config=noise_model_config))
                all_probs.append(probs)


        with tf.device('/cpu:0'):
            merged_probs = tf.concat(all_probs, axis=0)

        restore('gen', gen_ckpt)
        restore('dis', dis_ckpt)

        probs_under_model = get_seq_probs(seqs=inp,
                                          batch_size=batch_size_per_chunk * num_gpus,
                                          token_place_holders=all_tokens,
                                          num_gpus=num_gpus,
                                          tf_outputs=merged_probs,
                                          ignore_ids_np=ignore_ids_np,
                                          ignore_ids=ignore_ids)
        return probs_under_model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp')
    parser.add_argument('--noises')
    parser.add_argument('--noise-model-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--model-config', default='/private/home/kitsing/git/grover/lm/configs/base.json')
    parser.add_argument('--batch-size', default=8)
    parser.add_argument('--seq-length', default=1025)
    parser.add_argument('--num-gpus', default=8)
    parser.add_argument('--gen-ckpt', default='/checkpoint/kitsing/grover-models/base/model.ckpt')
    parser.add_argument('--dis-ckpt', default='/checkpoint/kitsing/grover-models/base/model.ckpt')
    args = parser.parse_args()
    from glob import glob
    noise_files = glob(args.noises)
    noise_tokens, noise_probs = get_all_noises(noise_files)
    noise_probs_under_model = compute_prob_under_model(noise_tokens, args.noise_model_config, args.model_config,
                                                       args.batch_size, args.num_gpus, args.seq_length, args.gen_ckpt,
                                                       args.dis_ckpt)
    probs = compute_nce_probs(args.inp, noise_probs, noise_probs_under_model)
    print(probs)
    greater_than_chance = probs > - np.log(probs.shape[1])
    print(greater_than_chance)
    print(np.sum(greater_than_chance))

if __name__ == '__main__':
    main()