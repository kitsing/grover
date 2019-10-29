import tensorflow as tf
import numpy as np
import sys
import argparse

sys.path.append('../')
from lm.modeling import GroverConfig, sample
from sample.encoder import get_encoder, format_context, _tokenize_article_pieces, extract_generated_target
from tqdm import tqdm
from os import environ
from random import seed as rnd_seed


def serialize(tokens: np.ndarray, probs_to_serialize: np.ndarray, prefix: str, dir: str):
    """

    :param tokens:
    :param probs_to_serialize:
    :param prefix:
    :param dir:
    :return:
    """
    from tempfile import mkstemp
    handle, filename = mkstemp(prefix=prefix, dir=dir, suffix='.npz')
    from os import close
    close(handle)
    np.savez(file=filename, tokens=tokens, probs=probs_to_serialize)


parser = argparse.ArgumentParser(description='Uncontextual generation')
parser.add_argument(
    '-model_config_fn',
    dest='model_config_fn',
    default='../lm/configs/base.json',
    type=str,
    help='Configuration JSON for the model',
)
parser.add_argument(
    '-model_ckpt',
    dest='model_ckpt',
    default='../models/base/model.ckpt',
    type=str,
    help='checkpoint file for the model',
)
parser.add_argument(
    '-batch_size',
    dest='batch_size',
    default=50,
    type=int,
    help='How many things to generate per context. will split into chunks if need be',
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=1,
    type=int,
    help='Number of folds. useful if we want to split up a big file into multiple jobs.',
)
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on. useful if we want to split up a big file into multiple jobs.'
)
parser.add_argument(
    '-max_batch_size',
    dest='max_batch_size',
    default=None,
    type=int,
    help='max batch size. You can leave this out and we will infer one based on the number of hidden layers',
)
parser.add_argument('--seed', default=42, type=int)

parser.add_argument('-prefix', default='unconditioned_', type=str)
parser.add_argument('-dir', default='./', type=str)
parser.add_argument('-num_gpus', default=8, type=int)

args = parser.parse_args()

seed = int(environ['SLURM_PROCID']) + args.seed
rnd_seed(seed)
tf.set_random_seed(seed)

encoder = get_encoder()
news_config = GroverConfig.from_json_file(args.model_config_fn)

print('start: {}'.format(encoder.__dict__['begin_article']))
print('end: {}'.format(encoder.__dict__['end_article']))

# We might have to split the batch into multiple chunks if the batch size is too large
default_mbs = {12: 32, 24: 16, 48: 3}
max_batch_size = args.max_batch_size if args.max_batch_size is not None else default_mbs[news_config.num_hidden_layers]

# factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
num_chunks = int(np.ceil(args.batch_size / max_batch_size))
batch_size_per_chunk = int(np.ceil(args.batch_size / num_chunks))
print("\n~~\nbatch size={}, max batch size={}, num chunks={}, batch size per chunk={}\n~~\n".format(
    args.batch_size, max_batch_size, num_chunks, batch_size_per_chunk), flush=True)

tf_config = tf.ConfigProto(allow_soft_placement=True)

with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
    initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
    p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
    eos_token = tf.placeholder(tf.int32, [])
    ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])

    all_tokens = []
    all_probs = []


    def pad_to_1025(tensor):
        tensor_length = tensor.shape[1]
        to_pad = tf.zeros((tensor.shape[0], 1025 - tensor_length), dtype=tensor.dtype)
        return tf.concat((tensor, to_pad), axis=1)

    for i in range(args.num_gpus):
        with tf.device('/gpu:'+str(i)):
            tokens, probs = sample(news_config=news_config, initial_context=initial_context,
                                   eos_token=eos_token, ignore_ids=ignore_ids, p_for_topp=p_for_topp,
                                   do_topk=False, seed=i)
            all_tokens.append(pad_to_1025(tokens))
            all_probs.append(probs)

    with tf.device('/cpu:0'):
        merged_tokens = tf.concat(all_tokens, axis=0)
        merged_probs = tf.concat(all_probs, axis=0)

    saver = tf.train.Saver()
    saver.restore(sess, args.model_ckpt)

    # Let's go!
    for batch in tqdm(range(num_chunks), disable=None):
        context_formatted = [encoder.__dict__['begin_article'],]

        # Indices we definitely DONT WANT TO PREDICT
        ignore_ids_np = np.array(encoder.special_tokens_onehot)
        ignore_ids_np[encoder.__dict__['end_article']] = 0

        tokens_out, probs_out = sess.run([merged_tokens, merged_probs],
                                         feed_dict={initial_context: [context_formatted] * batch_size_per_chunk,
                                                    eos_token: encoder.__dict__['end_article'],
                                                    ignore_ids: ignore_ids_np,
                                                    p_for_topp: np.ones((batch_size_per_chunk,),
                                                                        dtype=np.float32)})
        lengths = np.argmax(tokens_out[:, 1:] == encoder.__dict__['end_article'], axis=1) + 1
        mask = np.tile(np.arange(tokens_out.shape[1] - 1, dtype=np.int32)[np.newaxis, :],(tokens_out.shape[0], 1))
        masked = mask < lengths[:, np.newaxis]
        seq_probs = (masked * probs_out).sum(axis=1)
        serialize(tokens_out, seq_probs, args.prefix, args.dir)
