import tensorflow as tf
import numpy as np
import sys
import argparse

from nce.utils import restore, get_seq_probs

sys.path.append('../')
from lm.modeling import GroverConfig, eval_seq, sample
from sample.encoder import get_encoder
from tqdm import tqdm
from os import environ
from os.path import basename
from random import seed as rnd_seed
from math import ceil

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument(
    '--model-config-fn',
    default='../lm/configs/base.json',
    type=str,
    help='Configuration JSON for the model',
)
parser.add_argument(
    '--noise-model-config-fn',
    default='../lm/configs/base.json',
    type=str,
    help='Configuration JSON for the noise model',
)
parser.add_argument(
    '--gen-model-ckpt',
    default='../models/base/model.ckpt',
    type=str,
    help='Grover checkpoint file for the model',
)
parser.add_argument(
    '--dis-model-ckpt',
    default='',
    type=str,
    help='discriminator checkpoint file for the model',
)
parser.add_argument(
    '--noise-model-ckpt',
    default='',
    type=str,
    help='discriminator checkpoint file for the model',
)
parser.add_argument(
    '--batch-size',
    default=50,
    type=int,
    help='How many things to generate per context. will split into chunks if need be',
)
parser.add_argument('--num-noise-chunks', default=1, type=int)
parser.add_argument('--files', default='input.npz', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--correction-factor', default=1., type=float) # correction factor: 14136832/13628509
parser.add_argument('--seq-length', default=1025, type=int)
parser.add_argument('--output-path', default='./', type=str)
parser.add_argument('--noise-output-path', default='./', type=str)
parser.add_argument('--num-gpus', default=8, type=int)
parser.add_argument('--fixed-sample-size', default=-1, type=int)

args = parser.parse_args()
args.fold = int(environ['SLURM_PROCID'])
args.num_folds = int(environ['SLURM_NTASKS'])
from glob import glob

all_files = list(glob(args.files))[:]

args.num_files_per_node = int(ceil(len(all_files) / args.num_folds))
args.our_files = all_files[args.fold * args.num_files_per_node: (args.fold+1) * args.num_files_per_node]

# for training we are keeping the seed the same across all runs
seed = args.seed

rnd_seed(seed)
tf.set_random_seed(seed)

encoder = get_encoder()
news_config = GroverConfig.from_json_file(args.model_config_fn)
noise_news_config = GroverConfig.from_json_file(args.noise_model_config_fn)

print('start: {}'.format(encoder.__dict__['begin_article']))
print('end: {}'.format(encoder.__dict__['end_article']))

tf_config = tf.ConfigProto(allow_soft_placement=True)

context_formatted = [encoder.__dict__['begin_article'], ]


def output_noise():
    noise_token_chunks = []
    noise_prob_chunks = []
    for chunk in tqdm(range(args.num_noise_chunks), disable=None):
        noise_token_chunk, noise_prob_chunk = sess.run([merged_sampled_tokens, merged_sampled_probs],
                                                       feed_dict={
                                                           initial_context: [context_formatted] * batch_size_per_chunk,
                                                           eos_token: encoder.__dict__['end_article'],
                                                           ignore_ids: ignore_ids_np,
                                                           p_for_topp: np.ones((batch_size_per_chunk,),
                                                                               dtype=np.float32)}
                                                       )
        eos_positions = np.argmax(noise_token_chunk == encoder.__dict__['end_article'], axis=1)
        valid_seqs = (eos_positions != 0)
        mask = np.tile(np.arange(1025, dtype=np.int32)[None, :], (eos_positions.shape[0], 1))
        masked = mask <= eos_positions[:, None]
        noise_token_chunk = np.where(masked, noise_token_chunk, encoder.padding)[valid_seqs]
        noise_token_chunks.append(noise_token_chunk)

        prob_mask = masked[:, 1:]
        prob_masked = (prob_mask * noise_prob_chunk)[valid_seqs]
        noise_prob_masked = np.sum(prob_masked, axis=1)
        assert prob_masked.shape[0] == noise_token_chunk.shape[0]
        noise_prob_chunks.append(noise_prob_masked)
    noise_tokens = np.concatenate(noise_token_chunks, axis=0)
    n_probs = np.concatenate(noise_prob_chunks, axis=0)
    if args.fixed_sample_size > 0:
        assert noise_tokens.shape[0] > args.fixed_sample_size, noise_tokens.shape
        noise_tokens = noise_tokens[:args.fixed_sample_size]
        n_probs = n_probs[:args.fixed_sample_size]
    # evaluate the noise samples under our model
    noise_probs_under_model = get_seq_probs(seqs=noise_tokens,
                                            batch_size=args.batch_size * args.num_gpus,
                                            token_place_holders=all_tokens,
                                            num_gpus=args.num_gpus,
                                            tf_outputs=merged_probs,
                                            ignore_ids_np=ignore_ids_np,
                                            ignore_ids=ignore_ids, sess=sess,
                                            seq_length=args.seq_length)
    do_sanity_check = False
    if do_sanity_check:
        noise_probs_sanity_check = get_seq_probs(seqs=noise_tokens,
                                                 batch_size=args.batch_size * args.num_gpus,
                                                 token_place_holders=all_tokens,
                                                 num_gpus=args.num_gpus,
                                                 tf_outputs=merged_noise_probs,
                                                 ignore_ids_np=ignore_ids_np,
                                                 ignore_ids=ignore_ids, sess=sess, seq_length=args.seq_length)
        diff = noise_probs_sanity_check - n_probs
        print(f'sanity check: {np.sum(diff * diff)} {diff}')
    assert noise_probs_under_model.shape == n_probs.shape
    noise_output_fname = f'{args.noise_output_path}/{args.fold}.output.npz'
    np.savez(noise_output_fname, noise_probs_under_model=np.reshape(noise_probs_under_model, (-1,)),
             noise_probs_under_noise=np.reshape(n_probs, (-1,)))


with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
    batch_size_per_chunk = args.batch_size
    all_tokens = []
    all_probs = []
    all_noise_probs = []

    initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
    p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
    eos_token = tf.placeholder(tf.int32, [])
    ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])
    ignore_ids_np = np.array(encoder.special_tokens_onehot)
    ignore_ids_np[encoder.__dict__['end_article']] = 0
    for i in range(args.num_gpus):
        with tf.device('/gpu:' + str(i)):
            # actual examples
            tokens = tf.placeholder(tf.int32, [batch_size_per_chunk, args.seq_length])
            all_tokens.append(tokens)
            probs = tf.stop_gradient(eval_seq(news_config, tokens, args.correction_factor, baseline=False,
                                              ignore_ids=ignore_ids, gen_config=noise_news_config))
            all_probs.append(probs)

            noise_probs = tf.stop_gradient(eval_seq(noise_news_config,
                                                    tokens,
                                                    args.correction_factor,
                                                    baseline=True, gen_scope='newslm',
                                                    ignore_ids=ignore_ids))
            all_noise_probs.append(noise_probs)

    with tf.device('/cpu:0'):
        merged_probs = tf.concat(all_probs, axis=0)
        merged_noise_probs = tf.concat(all_noise_probs, axis=0)

    restore('gen', args.gen_model_ckpt, sess)
    restore('dis', args.dis_model_ckpt, sess)
    restore('newslm', args.noise_model_ckpt, sess)

    # get noise samples first
    # output_noise()

    # s_bar_noise = logsumexp(np.reshape(noise_probs_under_model, (-1,)) - np.reshape(n_probs, (-1,)), keepdims=True).reshape((-1,))
    # print(f's_bar_noise: {s_bar_noise} # of noise samples: {n_probs.shape}')

    # evaluate input tensors under both noise and our model
    from os.path import exists

    for file in args.our_files:
        output_fname = f'{args.output_path}/{basename(file)}.loss.npz'
        if exists(f'{output_fname}'):
            tf.logging.info(f'{output_fname} already exists. skipping...')
            continue
        with np.load(file) as loaded_numpy:
            all_seqs = loaded_numpy['tokens']
            input_probs_under_model = get_seq_probs(seqs=all_seqs,
                                                    batch_size=args.batch_size * args.num_gpus,
                                                    token_place_holders=all_tokens,
                                                    num_gpus=args.num_gpus, tf_outputs=merged_probs,
                                                    ignore_ids=ignore_ids,
                                                    ignore_ids_np=ignore_ids_np, sess=sess, seq_length=args.seq_length)

            input_probs_under_noise = get_seq_probs(seqs=all_seqs,
                                                    batch_size=args.batch_size * args.num_gpus,
                                                    token_place_holders=all_tokens,
                                                    num_gpus=args.num_gpus, tf_outputs=merged_noise_probs,
                                                    ignore_ids=ignore_ids,
                                                    ignore_ids_np=ignore_ids_np, sess=sess, seq_length=args.seq_length)

            np.savez(output_fname, input_probs_under_model=input_probs_under_model,
                     input_probs_under_noise=input_probs_under_noise)
