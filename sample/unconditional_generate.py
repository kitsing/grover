import tensorflow as tf
import numpy as np
import sys
import json

sys.path.append('../')
from lm.modeling import GroverModel, GroverConfig, sample
from sample.encoder import get_encoder, format_context, _tokenize_article_pieces, extract_generated_target
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Contextual generation (aka given some metadata we will generate articles')
parser.add_argument(
    '-out_fn',
    dest='out_fn',
    type=str,
    help='Out jsonl, which will contain the completed jsons',
)
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


args = parser.parse_args()

encoder = get_encoder()
news_config = GroverConfig.from_json_file(args.model_config_fn)

# We might have to split the batch into multiple chunks if the batch size is too large
default_mbs = {12: 32, 24: 16, 48: 3}
max_batch_size = args.max_batch_size if args.max_batch_size is not None else default_mbs[news_config.num_hidden_layers]

# factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
num_chunks = int(np.ceil(args.batch_size / max_batch_size))
batch_size_per_chunk = int(np.ceil(args.batch_size / num_chunks))
print("\n~~\nbatch size={}, max batch size={}, num chunks={}, batch size per chunk={}\n~~\n".format(
    args.batch_size, max_batch_size, num_chunks, batch_size_per_chunk), flush=True)

tf_config = tf.ConfigProto(allow_soft_placement=True)

with tf.Session(config=tf_config, graph=tf.Graph()) as sess, \
        open(args.out_fn, 'w') as f_out:
    initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
    p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
    eos_token = tf.placeholder(tf.int32, [])
    ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])
    tokens, probs = sample(news_config=news_config, initial_context=initial_context,
                           eos_token=eos_token, ignore_ids=ignore_ids, p_for_topp=p_for_topp,
                           do_topk=False)

    saver = tf.train.Saver()
    saver.restore(sess, args.model_ckpt)

    # Let's go!
    articles = [ {} ]
    for i, article in enumerate(tqdm(articles)):
        context_formatted = []
        context_formatted.append(encoder.__dict__['begin_article'])

        # Indices we definitely DONT WANT TO PREDICT
        ignore_ids_np = np.array(encoder.special_tokens_onehot)
        ignore_ids_np[encoder.__dict__['end_article']] = 0

        gens = []
        gens_raw = []
        gen_probs = []

        chunk_log_probs = []
        for chunk_i in range(num_chunks):
            tokens_out, probs_out = sess.run([tokens, probs],
                                             feed_dict={initial_context: [context_formatted] * batch_size_per_chunk,
                                                        eos_token: encoder.__dict__['end_article'],
                                                        ignore_ids: ignore_ids_np,
                                                        p_for_topp: np.ones((batch_size_per_chunk,),
                                                                            dtype=np.float32)})
            eos_position = np.argmax(tokens_out == encoder.__dict__['end_article'], axis=1) + 1
            mask = np.arange(tokens_out.shape[1], dtype=np.int32)[np.newaxis, :].tile((batch_size_per_chunk, 1))
            masked = mask < eos_position

            chunk_log_probs.append((masked * probs_out).sum(axis=1))
            for t_i, p_i in zip(tokens_out, probs_out):
                extraction = extract_generated_target(output_tokens=t_i, encoder=encoder, target='article')
                gens.append(extraction['extraction'])

                # NOTE: Originally I didn't add the +1 which meant that end article was being cut off. whoops.
                # better add that!
                gens_raw.append(t_i[extraction['start_ind']:extraction['end_ind'] + 1].tolist())

                assert extraction['start_ind'] == len(context_formatted)
                gen_probs.append(p_i[:extraction['end_ind'] - len(context_formatted) + 1].tolist())
        article['gens_log_probs'] = np.concatenate(chunk_log_probs, axis=0).tolist()
        for g_i, (g, g_raw) in enumerate(zip(gens, gens_raw)):
            article[f'gens_article_{g_i}'] = g
            article[f'gensraw_article_{g_i}'] = g_raw
        article['probs_article'] = gen_probs

        # these were in there for whatever reason...
        article.pop('input_ids_conditional', None)
        article.pop('input_ids_unconditional', None)
        f_out.write(json.dumps(article) + '\n')
