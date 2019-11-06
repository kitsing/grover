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
parser.add_argument('--file', default='input.npz', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--correction-factor', default=1., type=float) # correction factor: 14136832/13628509
parser.add_argument('--seq-length', default=1025, type=int)
parser.add_argument('--output-path', default='./', type=str)
parser.add_argument('--num-gpus', default=8, type=int)

args = parser.parse_args()
args.fold = int(environ['SLURM_PROCID'])
args.num_folds = int(environ['SLURM_NTASKS'])

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


def restore(scope, checkpoint):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    checkpoint_vars = tf.train.list_variables(checkpoint)
    checkpoint_names = [_[0] for _ in checkpoint_vars]
    assignment_map = dict()
    unused_vars_in_checkpoint = set(checkpoint_names)
    for var in vars:
        name = var.name
        assert name.endswith(':0')
        name = name[:-2]

        # hack
        if name.startswith('discriminator_final_layer'):
            if name in unused_vars_in_checkpoint:
                assignment_map[name] = var
            else:
                tf.logging.warn(f'key not found: {name}')
            continue

        splitted_name = name.split(scope)
        if len(splitted_name) > 1:
            new_name = ''.join(['newslm'] + splitted_name[1:])
            if new_name in unused_vars_in_checkpoint:
                assignment_map[new_name] = var
                tf.logging.info(f'key found: {new_name} -> {name}')
                unused_vars_in_checkpoint.remove(new_name)
            else:
                tf.logging.warn(f'key not found: {new_name}')
        else:
            tf.logging.warn(f'key {name} does not start with {scope}')

    tf.logging.warn(f'unused variables in checkpoint: {unused_vars_in_checkpoint}')
    # print(gen_assignment_map)
    saver = tf.train.Saver(var_list=assignment_map)
    saver.restore(sess, checkpoint)


context_formatted = [encoder.__dict__['begin_article'], ]


def get_seq_probs(seqs, batch_size, token_place_holders, num_gpus, tf_outputs):
    outputs = []
    num_batches = int(ceil(seqs.shape[0] / batch_size))
    for batch in tqdm(range(num_batches), disable=None):
        this_batch: np.ndarray = seqs[batch_size * batch: batch_size * (batch + 1)]
        feed_dict = {}
        # pad to fill all GPUs
        if this_batch.shape[0] < batch_size:
            to_append = np.zeros((batch_size - this_batch.shape[0],
                                  args.seq_length), dtype=this_batch.dtype)
            this_batch = np.concatenate((this_batch, to_append), axis=0)
        splitted_batch = np.split(this_batch, num_gpus)
        for tok, b in zip(token_place_holders, splitted_batch):
            feed_dict[tok] = b
        probs_out = sess.run([tf_outputs],
                             feed_dict=feed_dict)
        outputs.append(probs_out)
    return np.concatenate(outputs, axis=0).reshape((-1,))[:seqs.shape[0]]


with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
    batch_size_per_chunk = 8
    all_tokens = []
    all_probs = []
    all_noise_probs = []

    all_sampled_tokens = []
    all_sampled_probs = []

    initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
    p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
    eos_token = tf.placeholder(tf.int32, [])
    ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])
    ignore_ids_np = np.array(encoder.special_tokens_onehot)
    ignore_ids_np[encoder.__dict__['end_article']] = 0
    for i in range(args.num_gpus):
        with tf.device('/gpu:' + str(i)):
            # sampled noises
            sampled_tokens, sampled_probs = sample(news_config=news_config, initial_context=initial_context,
                                                   eos_token=eos_token, ignore_ids=ignore_ids, p_for_topp=p_for_topp,
                                                   do_topk=False, seed=i, max_out_tensor=True, vanilla=True)
            all_sampled_tokens.append(sampled_tokens)
            all_sampled_probs.append(sampled_probs)

            # actual examples
            tokens = tf.placeholder(tf.int32, [batch_size_per_chunk, args.seq_length])
            all_tokens.append(tokens)
            probs = tf.stop_gradient(eval_seq(news_config, tokens, args.correction_factor, baseline=False))
            all_probs.append(probs)

            noise_probs = tf.stop_gradient(eval_seq(noise_news_config,
                                                    tokens,
                                                    args.correction_factor,
                                                    baseline=True))
            all_noise_probs.append(noise_probs)

    with tf.device('/cpu:0'):
        merged_probs = tf.concat(all_probs, axis=0)
        merged_sampled_probs = tf.concat(all_sampled_probs, axis=0)
        merged_sampled_tokens = tf.concat(all_sampled_tokens, axis=0)
        merged_noise_probs = tf.concat(all_noise_probs, axis=0)

    restore('gen', args.gen_model_ckpt)
    restore('dis', args.dis_model_ckpt)
    restore('newslm', args.noise_model_ckpt)

    # get noise samples first
    noise_token_chunks = []
    noise_prob_chunks = []
    for chunk in tqdm(range(args.num_noise_chunks), disable=None):
        noise_token_chunk, noise_prob_chunk = sess.run([sampled_tokens, sampled_probs],
                                                       feed_dict={
                                                           initial_context: [context_formatted] * batch_size_per_chunk,
                                                           eos_token: encoder.__dict__['end_article'],
                                                           ignore_ids: ignore_ids_np,
                                                           p_for_topp: np.ones((batch_size_per_chunk,),
                                                                               dtype=np.float32)}
                                                       )
        eos_positions = np.argmax(noise_token_chunk == encoder.__dict__['end_article'], axis=1)
        mask = np.tile(np.arange(1025, dtype=np.int32)[None, :], (eos_positions.shape[0], 1))
        masked = mask <= eos_positions[:, None]
        noise_token_chunk = np.where(masked, noise_token_chunk, encoder.padding)
        noise_token_chunks.append(noise_token_chunk)

        prob_mask = masked[:, 1:]
        prob_masked = prob_mask * noise_prob_chunk
        noise_prob_masked = np.sum(prob_masked, axis=1)
        noise_prob_chunks.append(noise_prob_masked)

    noise_tokens = np.concatenate(noise_token_chunks, axis=0)
    noise_probs = np.concatenate(noise_prob_chunks, axis=0)

    # evaluate the noise samples under our model
    noise_probs_under_model = get_seq_probs(seqs=noise_tokens,
                                            batch_size=args.batch_size,
                                            token_place_holders=all_tokens,
                                            num_gpus=args.num_gpus,
                                            tf_outputs=merged_probs)
    assert noise_probs_under_model.shape == noise_probs.shape

    s_bar_noise = logsumexp(np.reshape(noise_probs_under_model, (-1,)) - np.reshape(noise_probs, (-1,)))

    # evaluate input tensors under both noise and our model
    from math import ceil
    from os.path import exists
    output_fname = f'{args.output_path}/{basename(args.file)}.out.npz'
    if exists(f'{output_fname}'):
        tf.logging.info(f'{output_fname} already exists. skipping...')
        exit(0)
    with np.load(args.file) as loaded_numpy:
        all_seqs = loaded_numpy['tokens']
        input_probs_under_model = get_seq_probs(seqs=all_seqs,
                                                batch_size=args.batch_size,
                                                token_place_holders=all_tokens,
                                                num_gpus=args.num_gpus, tf_outputs=merged_probs)

        input_probs_under_noise = get_seq_probs(seqs=all_seqs,
                                                batch_size=args.batch_size,
                                                token_place_holders=all_tokens,
                                                num_gpus=args.num_gpus, tf_outputs=merged_noise_probs)

        s_bar_real = input_probs_under_model - input_probs_under_noise
        s_bar_noise_expanded = np.tile(s_bar_noise[np.newaxis, :], (s_bar_real.shape[0], 1))
        classification_z = logsumexp(np.concatenate((s_bar_real, s_bar_noise_expanded), axis=1), axis=1)
        loss = -(s_bar_real - classification_z)
        output_fname = f'{args.output_path}/{basename(output_fname)}.loss.npz'
        np.savez(output_fname, loss=loss)
