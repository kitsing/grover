# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import tensorflow as tf


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t
    return example


def _decode_record_with_noise(record, noise, name_to_features, noise_name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t
    example['noises'] = tf.cast(noise, tf.int32)
    return example


def nce_input_fn_builder(input_files, noise_files, k,
                         seq_length,
                         is_training,
                         num_cpu_threads=8,
                         evaluate_for_fixed_number_of_steps=True,
                         input_batch_size=1, strategy=None
                         ):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    from sample.encoder import get_encoder
    encoder = get_encoder()
    end_symbol = encoder.__dict__['end_article']

    def build_gen(np_filenames, batch_size):
        import numpy as np

        def pad_along_axis(array: np.ndarray, target_length, axis=0):
            pad_size = target_length - array.shape[axis]
            axis_nb = len(array.shape)

            if pad_size <= 0:
                return array

            npad = [(0, 0) for x in range(axis_nb)]
            npad[axis] = (0, pad_size)

            b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

            return b

        def gen():
            fname_list = list(np_filenames)
            from random import shuffle
            shuffle(fname_list)
            remainder = []
            remainder_len = 0
            while len(fname_list) > 0:
                while remainder_len >= batch_size:
                    concat = np.concatenate(remainder, axis=0)
                    to_yield = concat[:batch_size]
                    remainder = [concat[batch_size:]]
                    remainder_len = remainder[0].shape[0]
                    yield to_yield
                np_filename = fname_list.pop()
                with np.load(np_filename) as loaded:
                    s = loaded['tokens']
                    s = s[(s == end_symbol).argmax(axis=1) > 0] # filter out rows where we cannot find an EOS symbol
                    np.random.shuffle(s)
                    s = pad_along_axis(s, seq_length + 1, 1)
                    truncated_num_of_rows = s.shape[0] - s.shape[0] % batch_size
                    # discard portions where we cannot make into a batch
                    remainder.append(s[truncated_num_of_rows:])
                    remainder_len = remainder_len + s.shape[0] % batch_size
                    if truncated_num_of_rows == 0:
                        continue
                    s = s[:truncated_num_of_rows]
                    # mask out symbols past EOS
                    mask = np.arange(s.shape[1])[None, :] <= (s == end_symbol).argmax(axis=1)[:, None]
                    masked: np.ndarray = s * mask
                    for b in range(int(s.shape[0] / batch_size)):
                        yield masked[b*batch_size:(b+1)*batch_size]

        return gen

    built_gen = build_gen(noise_files, k)

    def input_fn(params, input_context: tf.distribute.InputContext = None):
        """The actual input function."""
        # batch_size = params["batch_size"]
        batch_size = input_context.get_per_replica_batch_size(input_batch_size)
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length + 1], tf.int64),
        }

        noise_name_to_features = {
            'noises': tf.FixedLenFeature((k, seq_length + 1), tf.int64),
            'noise_probs': tf.FixedLenFeature((k,), dtype=tf.float32)
        }

        nd = tf.data.Dataset.from_generator(built_gen,
                                            tf.int64,
                                            output_shapes=tf.TensorShape([k, seq_length+1]))
        nd = nd.repeat()

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = parallel_interleave_shuffle(input_files, input_context=input_context)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # If we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            if evaluate_for_fixed_number_of_steps:
                d = d.repeat()

        # zip with the noise dataset
        d = tf.data.Dataset.zip((d, nd))

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record, noise: _decode_record_with_noise(record, noise,
                                                                name_to_features,
                                                                noise_name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    def parallel_interleave_shuffle(input_files, input_context = None):
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        if input_context is not None:
            d = d.shard(input_context.num_input_pipelines,
                        input_context.input_pipeline_id)
        d = d.repeat()
        d = d.shuffle(buffer_size=len(input_files))
        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(input_files))
        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        # d = d.shuffle(buffer_size=1000)
        return d

    return input_fn


def input_fn_builder(input_files,
                     seq_length,
                     is_training,
                     num_cpu_threads=4,
                     evaluate_for_fixed_number_of_steps=True):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length + 1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.data.experimental.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # If we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            if evaluate_for_fixed_number_of_steps:
                d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


#  ~~~~~~~~~~~~~~ This is for classification / AF ~~~~~~~~~~~~~~~~~~
def classification_convert_examples_to_features(
        examples, max_seq_length, batch_size, encoder, output_file, labels, pad_extra_examples=False,
        chop_from_front_if_needed=True):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    label_map = {label: i for i, label in enumerate(labels)}

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        # begin_summary is our [CLS] token
        tokens = example['ids'] + [encoder.begin_summary]

        if len(tokens) > max_seq_length:
            if chop_from_front_if_needed:
                tokens = tokens[-max_seq_length:]
            else:
                tokens = example['ids'][:(max_seq_length-1)] + [encoder.begin_summary]
        elif len(tokens) < max_seq_length:
            tokens.extend([encoder.padding] * (max_seq_length - len(tokens)))

        features = collections.OrderedDict()
        features['input_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=tokens))
        features['label_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label_map[example['label']]]))
        features['is_real_example'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    if pad_extra_examples:
        for x in range(len(examples) % batch_size):
            features = collections.OrderedDict()
            features['input_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[0]*max_seq_length))
            features['label_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
            features['is_real_example'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
    writer.close()


def classification_input_fn_builder(input_file, seq_length, is_training,
                                    drop_remainder,
                                    buffer_size=100):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=buffer_size)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn
