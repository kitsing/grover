import tensorflow as tf
import numpy as np
import sys
import argparse
from glob import glob

sys.path.append('../')
from lm.modeling import GroverConfig, eval_seq
from sample.encoder import get_encoder
from tqdm import tqdm
from os import environ
from os.path import basename
from random import seed as rnd_seed
from math import ceil

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument(
    '--model-config-fn',
    dest='model_config_fn',
    default='../lm/configs/base.json',
    type=str,
    help='Configuration JSON for the model',
)
parser.add_argument(
    '--noise-model-config-fn',
    dest='noise_model_config_fn',
    default='../lm/configs/base.json',
    type=str,
    help='Configuration JSON for the model',
)
parser.add_argument(
    '--gen-model-ckpt',
    default='../models/base/model.ckpt',
    type=str,
    help='Grover checkpoint file for the model',
)
parser.add_argument(
    '--dis-model-ckpt',
    dest='dis_model_ckpt',
    default='',
    type=str,
    help='discriminator checkpoint file for the model',
)
parser.add_argument(
    '--batch-size',
    dest='batch_size',
    default=50,
    type=int,
    help='How many things to generate per context. will split into chunks if need be',
)
parser.add_argument('--files', default='./*.npz', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--correction-factor', default=1., type=float) # correction factor: 14136832/13628509
parser.add_argument('--seq-length', default=1025, type=int)
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--output-path', default='./', type=str)
parser.add_argument('--num-gpus', default=8, type=int)

parser.add_argument('--load')

args = parser.parse_args()
args.fold = int(environ['SLURM_PROCID'])
args.num_folds = int(environ['SLURM_NTASKS'])

# for training we are keeping the seed the same across all runs
seed = args.seed

rnd_seed(seed)
tf.set_random_seed(seed)

files_to_open = sorted(glob(args.files))
files_chunk = int(ceil(len(files_to_open) / args.num_folds))
our_files = files_to_open[args.fold * files_chunk:(args.fold + 1) * files_chunk]

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


with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
    all_tokens = []

    all_probs = []
    ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])
    ignore_ids_np = np.array(encoder.special_tokens_onehot)
    ignore_ids_np[encoder.__dict__['end_article']] = 0
    for i in range(args.num_gpus):
        with tf.device('/gpu:' + str(i)):
            tokens = tf.placeholder(tf.int32, [args.batch_size // args.num_gpus, args.seq_length])
            all_tokens.append(tokens)
            probs = eval_seq(news_config, tokens, args.correction_factor, baseline=args.baseline, ignore_ids=ignore_ids,
                             gen_config=noise_news_config)
            all_probs.append(probs)

    with tf.device('/cpu:0'):
        merged_probs = tf.concat(all_probs, axis=0)

    restore('gen', args.gen_model_ckpt)
    if not args.baseline:
        restore('dis', args.dis_model_ckpt)

    # Let's go!
    for f in tqdm(our_files, disable=None):
        from math import ceil
        from os.path import exists
        output_fname = f'{args.output_path}/{basename(f)}.out.npz'
        if exists(f'{output_fname}'):
            tf.logging.info(f'{output_fname} already exists. skipping...')
            continue
        final_prob_outputs = []
        with np.load(f) as loaded_numpy:
            all_seqs = loaded_numpy['cloze']
            num_batches = int(ceil(all_seqs.shape[0] / args.batch_size))
            for batch in tqdm(range(num_batches), disable=None):
                this_batch: np.ndarray = all_seqs[args.batch_size * batch: args.batch_size * (batch + 1)]
                feed_dict = {ignore_ids: ignore_ids_np}
                # pad to fill all GPUs
                if this_batch.shape[0] < args.batch_size:
                    to_append = np.zeros((args.batch_size - this_batch.shape[0],
                                          args.seq_length), dtype=this_batch.dtype)
                    this_batch = np.concatenate((this_batch, to_append), axis=0)
                splitted_batch = np.split(this_batch, args.num_gpus)
                for tok, b in zip(all_tokens, splitted_batch):
                    feed_dict[tok] = b
                probs_out = sess.run([merged_probs],
                                     feed_dict=feed_dict)

                final_prob_outputs.append(probs_out)
        final_prob_tensor = np.concatenate(final_prob_outputs, axis=0).reshape((-1,))[:all_seqs.shape[0]]
        np.savez(output_fname, unnormalized_probs=final_prob_tensor)
