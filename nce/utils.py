from math import ceil

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from nce.gen_noise_samples import args


def restore(scope, checkpoint, sess):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    checkpoint_vars = tf.train.list_variables(checkpoint)
    checkpoint_names = [_[0] for _ in checkpoint_vars]
    assignment_map = dict()
    unused_vars_in_checkpoint = set(checkpoint_names)
    unused_vars_in_graph = set(vars)
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
            # new_name = ''.join(['newslm'] + splitted_name[1:])
            new_name = 'newslm'.join(splitted_name)
            if new_name in unused_vars_in_checkpoint:
                assignment_map[new_name] = var
                tf.logging.info(f'key found: {new_name} -> {name}')
                unused_vars_in_checkpoint.remove(new_name)
                unused_vars_in_graph.remove(var)
            else:
                tf.logging.warn(f'key not found: {new_name}')
        else:
            tf.logging.warn(f'key {name} does not start with {scope}')

    unused_relevant_vars = [_ for _ in unused_vars_in_checkpoint if 'adafactor' not in _ and 'Adafactor' not in _]
    tf.logging.warn(f'scope={scope}, checkpoint={checkpoint}, unused relevant variables in checkpoint: {unused_relevant_vars}')
    tf.logging.warn(
        f'scope={scope}, checkpoint={checkpoint}, unused variables in graph: {unused_vars_in_graph}')
    # print(gen_assignment_map)
    saver = tf.train.Saver(var_list=assignment_map)
    saver.restore(sess, checkpoint)


def get_seq_probs(seqs, batch_size, token_place_holders, num_gpus, tf_outputs, ignore_ids, ignore_ids_np, sess, seq_length):
    if not isinstance(tf_outputs, list):
        tf_outputs = [tf_outputs]
    outputs = []
    for _ in range(len(tf_outputs)):
        outputs.append([])
    num_batches = int(ceil(seqs.shape[0] / batch_size))
    for batch in tqdm(range(num_batches), disable=None):
        this_batch: np.ndarray = seqs[batch_size * batch: batch_size * (batch + 1)]
        feed_dict = {ignore_ids: ignore_ids_np}
        # pad to fill all GPUs
        if this_batch.shape[0] < batch_size:
            to_append = np.zeros((batch_size - this_batch.shape[0],
                                  seq_length), dtype=this_batch.dtype)
            this_batch = np.concatenate((this_batch, to_append), axis=0)
        splitted_batch = np.split(this_batch, num_gpus)
        for tok, b in zip(token_place_holders, splitted_batch):
            feed_dict[tok] = b

        probs_out = sess.run(tf_outputs,
                             feed_dict=feed_dict)
        for o,p in zip(outputs,probs_out):
            o.append(p)
    return tuple([np.concatenate(o, axis=0).reshape((-1,))[:seqs.shape[0]] for o in outputs])


def get_clean_noise_chunk(noise_token_chunk, eoa, padding, seq_length):
    eos_positions = np.argmax(noise_token_chunk == eoa, axis=1)
    valid_seqs = (eos_positions != 0)
    mask = np.tile(np.arange(seq_length, dtype=np.int32)[None, :], (eos_positions.shape[0], 1))
    masked = mask <= eos_positions[:, None]
    noise_token_chunk = np.where(masked, noise_token_chunk, padding)[valid_seqs]
    return noise_token_chunk, valid_seqs, masked