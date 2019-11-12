import tensorflow as tf
import numpy as np
import sys
import argparse
from glob import glob

sys.path.append('../')
from lm.modeling import GroverConfig, eval_seq, sample
from sample.encoder import get_encoder
from tqdm import tqdm
from os import environ
from os.path import basename
from random import seed as rnd_seed
from scipy.special import logsumexp
from math import ceil
from nce.calculate_nce_loss import eval_seq, get_seq_probs, restore

parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument(
    '--noise-model-config-fn',
    default='../lm/configs/base.json',
    type=str,
    help='Configuration JSON for the noise model',
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
noise_news_config = GroverConfig.from_json_file(args.noise_model_config_fn)

print('start: {}'.format(encoder.__dict__['begin_article']))
print('end: {}'.format(encoder.__dict__['end_article']))

tf_config = tf.ConfigProto(allow_soft_placement=True)

context_formatted = [encoder.__dict__['begin_article'], ]

with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
    batch_size_per_chunk = args.batch_size
    all_noise_probs = []

    all_sampled_tokens = []
    all_sampled_probs = []

    initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
    p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
    eos_token = tf.placeholder(tf.int32, [])
    ignore_ids = tf.placeholder(tf.bool, [noise_news_config.vocab_size])
    ignore_ids_np = np.array(encoder.special_tokens_onehot)
    ignore_ids_np[encoder.__dict__['end_article']] = 0
    for i in range(args.num_gpus):
        with tf.device('/gpu:' + str(i)):
            # sampled noises
            sampled_tokens, sampled_probs = sample(news_config=noise_news_config, initial_context=initial_context,
                                                   eos_token=eos_token, ignore_ids=ignore_ids, p_for_topp=p_for_topp,
                                                   do_topk=False, seed=i, max_out_tensor=True, vanilla=True)
            all_sampled_tokens.append(sampled_tokens)
            all_sampled_probs.append(sampled_probs)

    with tf.device('/cpu:0'):
        merged_sampled_probs = tf.concat(all_sampled_probs, axis=0)
        merged_sampled_tokens = tf.concat(all_sampled_tokens, axis=0)

    restore('newslm', args.noise_model_ckpt)

    # get noise samples first
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

    noise_output_fname = f'{args.noise_output_path}/{args.fold}.noise.npz'
    np.savez(noise_output_fname, noise_tokens=noise_tokens,
             noise_probs_under_noise=np.reshape(n_probs, (-1,)))