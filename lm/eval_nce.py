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

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument(
    '--model-config-fn',
    dest='model_config_fn',
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
parser.add_argument('--correction-factor', default=1., type=float)
parser.add_argument('--seq-length', default=1025, type=int)

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
files_chunk = len(files_to_open) // args.num_folds
our_files = files_to_open[args.fold * files_chunk:(args.fold + 1) * files_chunk]

encoder = get_encoder()
news_config = GroverConfig.from_json_file(args.model_config_fn)

print('start: {}'.format(encoder.__dict__['begin_article']))
print('end: {}'.format(encoder.__dict__['end_article']))

tf_config = tf.ConfigProto(allow_soft_placement=True)

with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
    tokens = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
    all_probs = []
    for i in range(args.num_gpus):
        with tf.device('/gpu:' + str(i)):
            probs = eval_seq(news_config, tokens, args.correction_factor)
            all_probs.append(probs)

    with tf.device('/cpu:0'):
        merged_probs = tf.concat(all_probs, axis=0)

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen')
    print(gen_vars)
    gen_assignment_map = dict()
    for var in gen_vars:
        name = var.name
        splitted_name = name.split('gen')
        tf.logging.info(f'found in gen_checkpoint: {name}')
        if len(splitted_name) > 1:
            new_name = ''.join(['newslm'] + splitted_name[1:])
            tf.logging.info(f'new name: {new_name}')
            gen_assignment_map[new_name] = var
    print(gen_assignment_map)
    saver = tf.train.Saver(var_list=gen_assignment_map)
    saver.restore(sess, args.gen_model_ckpt)

    dis_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dis')
    dis_assignment_map = dict()
    for var in dis_vars:
        name = var.name
        splitted_name = name.split('dis')
        tf.logging.info(f'found in gen_checkpoint: {name}')
        if len(splitted_name) > 1:
            new_name = ''.join(['newslm'] + splitted_name[1:])
            tf.logging.info(f'new name: {new_name}')
            dis_assignment_map[new_name] = var
    print(dis_assignment_map)
    saver = tf.train.Saver(var_list=dis_assignment_map)
    saver.restore(sess, args.dis_model_ckpt)

    # Let's go!
    for f in tqdm(our_files, disable=None):
        final_prob_outputs = []
        with np.load(f) as loaded_numpy:
            all_seqs = loaded_numpy['cloze']
            num_batches = all_seqs.shape[0] // args.batch_size
            for batch in tqdm(range(num_batches), disable=None):
                this_batch = all_seqs[args.batch_size * batch: args.batch_size * (batch + 1)]
                probs_out = sess.run([merged_probs],
                                     feed_dict={tokens: this_batch})

                final_prob_outputs.append(probs_out)
        final_prob_tensor = np.concatenate(final_prob_outputs, axis=0)
        output_fname = f'{args.output_path}/{basename(f)}.out.npz'
        np.savez(output_fname, unnormalized_probs=final_prob_tensor)
