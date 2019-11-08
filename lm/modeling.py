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

import copy
import json
import math

import six
import tensorflow as tf

from lm import optimization_adafactor
from lm.utils import get_assignment_map_from_checkpoint, get_shape_list, get_attention_mask, gelu, layer_norm, dropout, \
    construct_scalar_host_call, get_inverted_mask
from typing import Optional

class GroverConfig(object):
    """Configuration for `GroverModel`"""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 initializer_range=0.02,
                 mask_padding: bool = False,
                 word_dropout_prob: float = 0.1,
                 final_projection_layer: bool = False,
                 reuse_gen: bool = False,
                 additional_transformer_layers: int = 4,
                 truncate_right: bool = False,
                 regularize_g: bool = False,
                 scope_prefix='newslm'):
        """Constructs NewsConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `GroverModel`.
          hidden_size: Size of the layers
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = 0
        self.scope_prefix=scope_prefix
        self.mask_padding = mask_padding
        self.word_dropout_prob = word_dropout_prob
        self.final_projection_layer = final_projection_layer
        self.reuse_gen = reuse_gen
        self.additional_transformer_layers = additional_transformer_layers
        self.truncate_right = truncate_right
        self.regularize_g = regularize_g

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `NewsConfig` from a Python dictionary of parameters."""
        config = GroverConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `NewsConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def __str__(self):
        return str(self.__dict__)


def mask_attention_for_ltr(attention_scores, attention_mask):
    """
    Mask attention so that we're only predicting going forward
    :param attention_scores: [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    :param attention_mask [query_length, key_length]
    :return: masked attention
    """
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    mask = attention_mask[None, None]
    return attention_scores * mask - tf.cast(1e10, attention_scores.dtype) * (1 - mask)


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def _attention_projection_and_transpose(x_flat, batch_size, seq_length, num_attention_heads, size_per_head,
                                        name, initializer_range=0.02):
    """
    :param x_flat: [batch_size*seq_length, width]
    :return: A fixed up tensor of size [batch_size, num_attention_heads, seq_length, size_per_head]
    """
    batch_size_seq_length, dim = get_shape_list(x_flat, expected_rank=2)

    # Had to remove this bc of generation script
    # if (batch_size_seq_length != batch_size * seq_length):
    #     raise ValueError("passed in a tensor of shape {} when batch_size={} and seq_length={}".format(
    #         (batch_size_seq_length, dim), batch_size, seq_length
    #     ))

    if dim != size_per_head * num_attention_heads:
        raise ValueError("passed in a tensor of shape {} when size_per_head={} and num_attention_heads={}".format(
            (batch_size_seq_length, dim), size_per_head, num_attention_heads
        ))

    projected = tf.layers.dense(
        x_flat,
        num_attention_heads * size_per_head,
        name=name,
        kernel_initializer=create_initializer(initializer_range))

    projected = tf.reshape(
        projected, [batch_size, seq_length, num_attention_heads, size_per_head])
    output_tensor = tf.transpose(projected, [0, 2, 1, 3])
    return output_tensor


def attention_layer(x_flat, attention_mask, batch_size, seq_length, size_per_head=512, num_attention_heads=1, *,
                    cache=None,
                    initializer_range=0.02, hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1, do_cache=False):
    """

    :param x_flat: Tensor input, should be [batch_size*seq_length, dim]
    :param attention_mask: Attention mask to use of size [seq_length, seq_length+cached_length]
    :param size_per_head: dim = size_per_head * num_attention_heads
    :param num_attention_heads:  dim = size_per_head * num_attention_heads
    :param cache: Optionally some past (cached) things of size
                [batch, 2, heads, sequence, features], where 2 is [k, v]
    :param do_cache: True if we should return cache
    :return: A new tensor of shape [batch_size, seq_length, dim]
    as well as a new cache "cached_keys_and_values" that will be of size
                                   [batch_size, 2, num_attention_heads, seq_length, dim]
    """
    batch_size_seq_length, dim = get_shape_list(x_flat, expected_rank=2)

    # Had to remove this because of generation script
    # if (batch_size_seq_length != batch_size * seq_length):
    #     raise ValueError("passed in a tensor of shape {} when batch_size={} and seq_length={}".format(
    #         (batch_size_seq_length, dim), batch_size, seq_length
    #     ))

    if dim != size_per_head * num_attention_heads:
        raise ValueError("passed in a tensor of shape {} when size_per_head={} and num_attention_heads={}".format(
            (batch_size_seq_length, dim), size_per_head, num_attention_heads
        ))

    # if do_cache and past is not None:
    #     Shape will be (batch_size, 2, num_attention_heads, past_seq_length, dim)
    #     past_shape = get_shape_list(past, 5)
    #     desired_shape = (batch_size, 2, num_attention_heads, seq_length, dim)
    #     if tuple(past_shape) != desired_shape:
    #         raise ValueError(f"The shape of the cache is {past_shape} but we want {desired_shape}")

    # [ batch_size, num_attention_heads, seq_length, size_per_head]
    query = _attention_projection_and_transpose(x_flat, batch_size=batch_size, seq_length=seq_length,
                                                num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                                name='query_layer',
                                                initializer_range=initializer_range)
    key = _attention_projection_and_transpose(x_flat, batch_size=batch_size, seq_length=seq_length,
                                              num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                              name='key_layer',
                                              initializer_range=initializer_range)

    value = _attention_projection_and_transpose(x_flat, batch_size=batch_size, seq_length=seq_length,
                                                num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                                name='value_layer',
                                                initializer_range=initializer_range)

    # Add to cache
    cached_keys_and_values = tf.stack([key, value], axis=1) if do_cache else None

    # Things that were relevant from the cache
    if cache is not None:
        pk, pv = tf.unstack(cache, axis=1)
        key = tf.concat([pk, key], axis=-2)
        value = tf.concat([pv, value], axis=-2)

    # Multiply [batch_size, num_attention_heads, seq_length, size_per_head] with
    #          [batch_size, num_attention_heads, size_per_head, seq_length+cached_length] ->
    #          [batch_size, num_attention_heads, seq_length, seq_length+cached_length]
    attention_scores = tf.matmul(query, key, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))
    attention_scores = mask_attention_for_ltr(attention_scores, attention_mask)
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    # NOPENOPENOPENOPE
    # attention_probs = factoreddropout(attention_probs, attention_probs_dropout_prob)

    # Multiply [batch_size, num_attention_heads, seq_length, seq_length+cached_length] with
    #          [batch_size, num_attention_heads, seq_length+cached_length, size_per_head] ->
    #          [batch_size, num_attention_heads, seq_length, size_per_head] ->
    context_layer = tf.matmul(attention_probs, value)

    # `context_layer` = [batch_size, seq_length, num_attention_heads, size_per_head]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    context_layer = tf.reshape(context_layer, [batch_size * seq_length, num_attention_heads * size_per_head])

    context_layer_projected = tf.layers.dense(
        context_layer,
        num_attention_heads * size_per_head,
        kernel_initializer=create_initializer(initializer_range),
        name='context_projection_layer'
    )
    context_layer_projected = dropout(context_layer_projected, hidden_dropout_prob)

    return context_layer_projected, cached_keys_and_values


def residual_mlp_layer(x_flat, intermediate_size, initializer_range=0.02, hidden_dropout_prob=0.1):
    """
    :param x: The attention output. It should be [batch_size*seq_length, dim]
    :param intermediate_size: the hidden projection. By default this is the input_dim * 4.

    in the original GPT we would return layer_norm(x_norm + h1) rather than layer_norm(x + h1)

    :return:
    """
    batch_size_seq_length, hidden_size = get_shape_list(x_flat, expected_rank=2)
    x_norm = layer_norm(x_flat, name='mlp_ln0')

    intermediate_output = tf.layers.dense(
        x_norm,
        intermediate_size,
        activation=gelu,
        kernel_initializer=create_initializer(initializer_range),
        name='intermediate',
    )

    output_for_residual = tf.layers.dense(
        intermediate_output,
        hidden_size,
        name='output',
        kernel_initializer=create_initializer(initializer_range))
    output_for_residual = dropout(output_for_residual, hidden_dropout_prob)

    layer_output = layer_norm(x_flat + output_for_residual, name='mlp_ln1')
    return layer_output


def residual_mlp_no_ln_layer(x_flat, intermediate_size, initializer_range=0.02, hidden_dropout_prob=0.1):
    """
    :param x: The attention output. It should be [batch_size*seq_length, dim]
    :param intermediate_size: the hidden projection. By default this is the input_dim * 4.

    in the original GPT we would return layer_norm(x_norm + h1) rather than layer_norm(x + h1)

    :return:
    """
    batch_size_seq_length, hidden_size = get_shape_list(x_flat, expected_rank=2)

    intermediate_output = tf.layers.dense(
        x_flat,
        intermediate_size,
        activation=gelu,
        kernel_initializer=create_initializer(initializer_range),
        name='intermediate',
    )

    output_for_residual = tf.layers.dense(
        intermediate_output,
        hidden_size,
        name='output',
        kernel_initializer=create_initializer(initializer_range))
    output_for_residual = dropout(output_for_residual, hidden_dropout_prob)
    return output_for_residual


def embed(input_ids,
          vocab_size,
          embedding_size,
          position_offset=0,
          initializer_range=0.02,
          max_position_embeddings=512,
          use_one_hot_embeddings=True):
    """reur and position embeddings
    :param input_ids: int Tensor of shape [batch_size, seq_length].
    :param vocab_size: number of words in vocab
    :param embedding_size: dimensionality of the embedding
    :param position_offset: aka number of cached tokens.
    :param initializer_range: float. Range of the weight initialization.
    :param max_position_embeddings: int. Maximum sequence length.
    :param use_one_hot_embeddings: probably want this to be true
    :return: [batch_size, seq_length, embedding_size] embedded tensor
    """
    (batch_size, seq_length) = get_shape_list(input_ids, expected_rank=2)

    embedding_table = tf.get_variable(
        name='word_embed',
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range),
    )

    assert_op = tf.assert_less_equal(tf.reduce_max(input_ids), vocab_size - 1)
    with tf.control_dependencies([assert_op]):
        if use_one_hot_embeddings:
            flat_input_ids = tf.reshape(input_ids, [-1])
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
            output_flat = tf.matmul(one_hot_input_ids, embedding_table)
        else:
            output_flat = tf.nn.embedding_lookup(embedding_table, input_ids)

        embedded_input = tf.reshape(output_flat, [batch_size, seq_length, embedding_size])

    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)

    with tf.control_dependencies([assert_op]):
        full_position_embeddings = tf.get_variable(
            name='pos_embed',
            shape=[max_position_embeddings, embedding_size],
            initializer=create_initializer(initializer_range),
        )
        # Since the position embedding table is a learned variable, we create it
        # using a (long) sequence length `max_position_embeddings`. The actual
        # sequence length might be shorter than this, for faster training of
        # tasks that do not have long sequences.
        #
        # So `full_position_embeddings` is effectively an embedding table
        # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
        # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
        # perform a slice.
        if position_offset == 0:
            embedded_input += tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])[None]
        else:
            # Tensorflow is too stupid to allow slicing
            flat_pos_ids = (tf.range(seq_length, dtype=tf.int32) + position_offset)
            one_hot_pos_ids = tf.one_hot(flat_pos_ids, depth=max_position_embeddings)

            # [seq_length, full_position_embeddings], [full_position_embeddings, dim]
            seq_embeds = tf.matmul(one_hot_pos_ids, full_position_embeddings)
            embedded_input += seq_embeds[None]

            # embedded_input += tf.slice(full_position_embeddings[position_offset:], [0, 0], [seq_length, -1])[None]

    return layer_norm(embedded_input, name='embed_norm'), embedding_table


def _top_p_sample(logits, ignore_ids=None, num_samples=1, p=0.9, seed: Optional[int] = None,
                  vanilla: bool = False):
    """
    Does top-p sampling. if ignore_ids is on, then we will zero out those logits.
    :param logits: [batch_size, vocab_size] tensor
    :param ignore_ids: [vocab_size] one-hot representation of the indices we'd like to ignore and never predict,
                        like padding maybe
    :param p: topp threshold to use, either a float or a [batch_size] vector
    :return: [batch_size, num_samples] samples

    # TODO FIGURE OUT HOW TO DO THIS ON TPUS. IT'S HELLA SLOW RIGHT NOW, DUE TO ARGSORT I THINK
    """
    with tf.variable_scope('top_p_sample'):
        batch_size, vocab_size = get_shape_list(logits, expected_rank=2)

        masked_out_logits = logits if ignore_ids is None else logits - tf.cast(ignore_ids[None], tf.float32) * 1e10
        probs = tf.nn.log_softmax(masked_out_logits,
                                  axis=-1)

        if vanilla or (isinstance(p, float) and p > 0.999999):
            # Don't do top-p sampling in this case
            # print("Top-p sampling DISABLED", flush=True)
            return {
                'probs': probs,
                'sample': tf.random.categorical(
                    logits=masked_out_logits,
                    num_samples=num_samples, dtype=tf.int32, seed=seed),
            }

        # [batch_size, vocab_perm]
        indices = tf.argsort(probs, direction='DESCENDING')
        cumulative_probabilities = tf.math.cumsum(tf.batch_gather(probs, indices), axis=-1, exclusive=False)

        # find the top pth index to cut off. careful we don't want to cutoff everything!
        # result will be [batch_size, vocab_perm]
        p_expanded = p if isinstance(p, float) else p[:, None]
        exclude_mask = tf.logical_not(
            tf.logical_or(cumulative_probabilities < p_expanded, tf.range(vocab_size)[None] < 1))

        # OPTION A - sample in the sorted space, then unsort.
        logits_to_use = tf.batch_gather(logits, indices) - tf.cast(exclude_mask, tf.float32) * 1e10
        sample_perm = tf.random.categorical(logits=logits_to_use, num_samples=num_samples)
        sample = tf.batch_gather(indices, sample_perm)

        # OPTION B - unsort first - Indices need to go back to 0 -> N-1 -- then sample
        # unperm_indices = tf.argsort(indices, direction='ASCENDING')
        # include_mask_unperm = tf.batch_gather(include_mask, unperm_indices)
        # logits_to_use = logits - (1 - tf.cast(include_mask_unperm, tf.float32)) * 1e10
        # sample = tf.random.categorical(logits=logits_to_use, num_samples=num_samples, dtype=tf.int32)

    return {
        'probs': probs,
        'sample': sample,
    }


def _top_k_sample(logits, ignore_ids=None, num_samples=1, k=10):
    """
    Does top-k sampling. if ignore_ids is on, then we will zero out those logits.
    :param logits: [batch_size, vocab_size] tensor
    :param ignore_ids: [vocab_size] one-hot representation of the indices we'd like to ignore and never predict,
                        like padding maybe
    :param p: topp threshold to use, either a float or a [batch_size] vector
    :return: [batch_size, num_samples] samples

    # TODO FIGURE OUT HOW TO DO THIS ON TPUS. IT'S HELLA SLOW RIGHT NOW, DUE TO ARGSORT I THINK
    """
    with tf.variable_scope('top_p_sample'):
        batch_size, vocab_size = get_shape_list(logits, expected_rank=2)

        probs = tf.nn.softmax(logits if ignore_ids is None else logits - tf.cast(ignore_ids[None], tf.float32) * 1e10,
                              axis=-1)
        # [batch_size, vocab_perm]
        indices = tf.argsort(probs, direction='DESCENDING')

        # find the top pth index to cut off. careful we don't want to cutoff everything!
        # result will be [batch_size, vocab_perm]
        k_expanded = k if isinstance(k, int) else k[:, None]
        exclude_mask = tf.range(vocab_size)[None] >= k_expanded

        # OPTION A - sample in the sorted space, then unsort.
        logits_to_use = tf.batch_gather(logits, indices) - tf.cast(exclude_mask, tf.float32) * 1e10
        sample_perm = tf.random.categorical(logits=logits_to_use, num_samples=num_samples)
        sample = tf.batch_gather(indices, sample_perm)

    return {
        'probs': probs,
        'sample': sample,
    }


class GroverModelResidual(object):
    def __init__(self,
                 config: GroverConfig,
                 is_training,
                 input_ids,
                 noises,
                 cache=None,
                 do_cache=False,
                 pad_token_id=0,
                 scope=None,
                 reuse=False, alpha: float = 1.,
                 ignore_noise: bool = False):
        """
        :param config:
        :param is_training:
        :param input_ids: Tensor thats of size [batch_size, seq_length]
        :param noises: Tensor that's of size [batch_size, k, seq_length]
        :param cache: Optionally, a tensor to use that will contain cached information of the size
            [batch_size, num_layers, 2, num_heads, cache_length, features]
        :param do_cache: Whether to cache again.
        :param pad_token_id: Which token will be used for padding (probably 0.)
        :param scope: scope to run this on
        """
        self.config = copy.deepcopy(config)
        self.is_training = is_training
        self.pad_token_id = pad_token_id

        self.alpha = alpha

        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_probs_dropout_prob = 0.0

        # autoencoder-like architecture
        original_batch_size = input_ids.shape[0]

        if self.config.truncate_right:
            input_ids = input_ids[:, -1]
        else:
            input_ids = input_ids[:, 1:]

        self.input_ids = input_ids
        self.batch_size, self.seq_length = get_shape_list(self.input_ids, 2)

        if not ignore_noise:
            self.k = noises.shape[1]
            if self.config.truncate_right:
                noises = noises[:, :, -1]
            else:
                noises = noises[:, :, 1:]
            self.noises = tf.reshape(noises, (-1, noises.shape[2]))
            self.noise_batch_size, _ = get_shape_list(self.noises, 2)
        else:
            self.k = 0

        if is_training and self.config.word_dropout_prob > 0.:
            from tensorflow.distributions import Bernoulli
            b = Bernoulli(probs=(1 - self.config.word_dropout_prob), dtype=tf.bool)
            word_dropout_mask = b.sample(sample_shape=(self.batch_size, self.seq_length))
            self.input_ids = tf.where(word_dropout_mask,
                                      self.input_ids,
                                      tf.fill(
                                          (self.batch_size, self.seq_length),
                                          self.pad_token_id)
                                      )
            if not ignore_noise:
                noise_dropout_mask = b.sample(sample_shape=(self.noise_batch_size, self.seq_length))
                self.noises = tf.where(noise_dropout_mask,
                                       self.noises,
                                       tf.fill(
                                           (self.noise_batch_size, self.seq_length),
                                           self.pad_token_id)
                                       )

        if self.config.mask_padding:
            label_weights = tf.cast(tf.not_equal(tf.reshape(self.input_ids, [-1]),
                                                 self.pad_token_id),
                                    dtype=tf.float32)
            if not ignore_noise:
                noise_label_weights = tf.cast(tf.not_equal(tf.reshape(self.noises, [-1]),
                                                 self.pad_token_id),
                                              dtype=tf.float32)
        else:
            label_weights = tf.reshape(tf.ones_like(self.input_ids, dtype=tf.float32), (-1,))
            if not ignore_noise:
                noise_label_weights = tf.reshape(tf.ones_like(self.noises, dtype=tf.float32), (-1,))

        assert config.max_position_embeddings >= self.seq_length
        caches = [None] * config.num_hidden_layers
        self.cache_length = 0

        with tf.variable_scope(default_name='newslm', reuse=reuse,
                               name_or_scope=scope):
            input_hidden_state, residuals = self.score_seq(caches, config, do_cache, is_training, label_weights, self.input_ids,
                                                           reuse, self.batch_size)
            self.hidden_state = input_hidden_state
            self.residuals = tf.reshape(residuals, (self.batch_size, 1))
            if not ignore_noise:
                noise_hidden_state, noise_residuals = self.score_seq(caches, config, do_cache, is_training, noise_label_weights,
                                                                     self.noises,
                                                                     reuse, self.noise_batch_size)
                self.noise_hidden_state = tf.reshape(noise_hidden_state, (self.batch_size, self.k, -1))
                self.noise_residuals = tf.reshape(noise_residuals, (self.batch_size, self.k))

        # THE OUTPUT BIAS DOES NOT SPARK JOY
        # output_bias = tf.get_variable('output_bias', shape=[config.vocab_size], initializer=tf.zeros_initializer())
        # self.logits_flat = tf.nn.bias_add(self.logits_flat, output_bias)

    def score_seq(self, caches, config, do_cache, is_training, label_weights, to_score,
                  reuse, original_batch_size):
        with tf.variable_scope("embeddings"):
            embeddings, embedding_table = embed(to_score, config.vocab_size,
                                                config.hidden_size,
                                                position_offset=self.cache_length,
                                                initializer_range=config.initializer_range,
                                                max_position_embeddings=config.max_position_embeddings,
                                                use_one_hot_embeddings=True)
        if self.config.reuse_gen:
            # reusing the original model for embeddings. to ensure correctness we make use of the original masks
            mask = get_attention_mask(self.seq_length, self.seq_length + self.cache_length, dtype=embeddings.dtype)
        else:
            # we don't really need masking
            # mask = get_inverted_mask(self.seq_length, self.seq_length + self.cache_length, dtype=embeddings.dtype)
            mask = tf.ones((self.seq_length, self.seq_length + self.cache_length), dtype=embeddings.dtype)
        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        hidden_state = tf.reshape(embeddings, [original_batch_size * self.seq_length, self.config.hidden_size])
        new_kvs = []
        for layer_idx, layer_cache in enumerate(caches):
            with tf.variable_scope('layer{:02d}'.format(layer_idx)):
                # [batch_size * seq_length, hidden_size]
                attention_output, new_kv = attention_layer(
                    hidden_state,
                    mask,
                    batch_size=original_batch_size,
                    seq_length=self.seq_length,
                    size_per_head=config.hidden_size // config.num_attention_heads,
                    num_attention_heads=config.num_attention_heads,
                    initializer_range=config.initializer_range,
                    hidden_dropout_prob=self.config.hidden_dropout_prob,
                    attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
                    do_cache=do_cache,
                    cache=layer_cache,
                )
                new_kvs.append(new_kv)

                # [batch_size * seq_length, hidden_size]
                hidden_state = residual_mlp_no_ln_layer(hidden_state + attention_output,
                                                        intermediate_size=config.intermediate_size,
                                                        hidden_dropout_prob=self.config.hidden_dropout_prob)
        if self.config.reuse_gen:
            raise NotImplementedError
            # have to stop gradient here as we don't want to finetune the generator (or do we?)
            hidden_state = tf.stop_gradient(hidden_state) * label_weights[:, None]
            # start some additional layers
            if self.config.additional_transformer_layers > 0:
                with tf.variable_scope("additional_embeddings"):
                    add_embeddings, \
                    add_embedding_table = embed(to_score, config.vocab_size,
                                                config.hidden_size,
                                                position_offset=self.cache_length,
                                                initializer_range=config.initializer_range,
                                                max_position_embeddings=config.max_position_embeddings,
                                                use_one_hot_embeddings=True)
                emb_shape = [original_batch_size * self.seq_length, self.config.hidden_size]
                additional_emb = tf.reshape(add_embeddings, emb_shape)
                hidden_state = tf.concat((hidden_state, additional_emb), axis=1)
                # we are no longer subject to the left-context limitation so we have a full mask here
                full_mask = tf.ones((self.seq_length, self.seq_length + self.cache_length), dtype=embeddings.dtype)

                for additional_layer_idx in range(self.config.additional_transformer_layers):
                    with tf.variable_scope('additional_layer{:02d}'.format(additional_layer_idx)):
                        # [batch_size * seq_length, hidden_size]
                        attention_output, new_kv = attention_layer(
                            hidden_state,
                            full_mask,
                            batch_size=original_batch_size,
                            seq_length=self.seq_length,
                            size_per_head=config.hidden_size * 2 // config.num_attention_heads,
                            num_attention_heads=config.num_attention_heads,
                            initializer_range=config.initializer_range,
                            hidden_dropout_prob=self.config.hidden_dropout_prob,
                            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
                            do_cache=do_cache,
                            cache=layer_cache,
                        )
                        new_kvs.append(new_kv)

                        # [batch_size * seq_length, hidden_size]
                        hidden_state = residual_mlp_layer(hidden_state + attention_output,
                                                          intermediate_size=config.intermediate_size,
                                                          hidden_dropout_prob=self.config.hidden_dropout_prob)
                hidden_state = tf.layers.dense(tf.layers.dropout(hidden_state,
                                                                 rate=self.config.hidden_dropout_prob,
                                                                 training=is_training),
                                               self.config.hidden_size, name='additional_final_layer',
                                               )
                hidden_state = layer_norm(hidden_state, name='layer_normed_final_layer')
        if config.final_projection_layer:
            raise NotImplementedError
            residuals = tf.layers.dense(tf.layers.dropout(hidden_state, rate=config.hidden_dropout_prob,
                                                          training=is_training),
                                        1, name='discriminator_final_layer', use_bias=False,
                                        reuse=reuse)
            residuals = tf.reshape(tf.reshape(residuals, (-1,)) * label_weights,
                                        (original_batch_size, self.seq_length)
                                        )
            residuals = tf.reduce_sum(residuals, axis=1)
        else:
            logits_flat = tf.matmul(hidden_state, embedding_table, transpose_b=True)
            to_score_flat = tf.reshape(to_score, (-1,))
            one_hot_labels = tf.one_hot(to_score_flat,
                                        depth=self.config.vocab_size,
                                        dtype=logits_flat.dtype)
            logprobs_flat = tf.nn.log_softmax(logits_flat, axis=-1)
            selected_and_masked = label_weights * tf.reduce_sum(logprobs_flat * one_hot_labels, axis=[-1])
            residuals = tf.reduce_sum(tf.reshape(selected_and_masked, (-1, self.seq_length)), axis=[-1])

        return hidden_state, residuals

    def reg_g_loss(self):
        noises = self.noise_residuals
        log_sum_exp = tf.reduce_logsumexp(noises, axis=1)
        diff = log_sum_exp - tf.log(tf.cast(noises.shape[1], tf.float32))
        return tf.reduce_mean(diff * diff, axis=0)

    def total_loss(self):
        if self.alpha == 1.:
            full_z = tf.concat((self.residuals, self.noise_residuals), axis=1)
            return - tf.reduce_mean(tf.nn.log_softmax(full_z, axis=1)[:, 0])
            # total_loss = - (self.residuals[:, 0] - tf.reduce_logsumexp(full_z, axis=1))
            # return tf.reduce_mean(total_loss, axis=0)
        else:
            # TODO implement this!
            raise NotImplementedError

    def pooled_output(self, clf_token):
        """
        Extract pooled output given a token that says where we should look
        :param clf_token:
        :return:
        """
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(self.input_ids, clf_token), tf.float32), 1), tf.int32)
        return tf.gather(self.hidden_state, tf.range(self.batch_size, dtype=tf.int32) * self.seq_length + pool_idx)


class GroverModel(object):
    def __init__(self,
                 config: GroverConfig,
                 is_training,
                 input_ids,
                 cache=None,
                 do_cache=False,
                 pad_token_id=0,
                 chop_off_last_token=True,
                 scope=None,
                 reuse=False):
        """
        :param config:
        :param is_training:
        :param input_ids: Tensor thats of size [batch_size, seq_length]
        :param cache: Optionally, a tensor to use that will contain cached information of the size
            [batch_size, num_layers, 2, num_heads, cache_length, features]
        :param do_cache: Whether to cache again.
        :param pad_token_id: Which token will be used for padding (probably 0.)
        :param chop_off_last_token: True if we will end up using this for TRAINING only. False if we want to generate.
                                    it means the last token in input_ids will not be processed by the model as input
        :param scope: scope to run this on
        """
        self.config = copy.deepcopy(config)
        self.is_training = is_training
        self.pad_token_id = pad_token_id

        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_probs_dropout_prob = 0.0

        if chop_off_last_token:
            self.target_ids = input_ids[:, 1:]
            self.input_ids = input_ids[:, :-1]
        else:
            self.input_ids = input_ids
            self.target_ids = tf.concat((input_ids[:, 1:],
                                         tf.constant(self.pad_token_id, dtype=self.input_ids.dtype,
                                                     shape=[get_shape_list(self.input_ids, 2)[0], 1])), 1)

        self.batch_size, self.seq_length = get_shape_list(self.input_ids, 2)

        if cache is None:
            caches = [None] * config.num_hidden_layers
            self.cache_length = 0
        else:
            batch_size_, num_layers_, two_, num_heads_, self.cache_length, features_ = get_shape_list(
                cache, expected_rank=6)
            assert batch_size_ == self.batch_size
            assert num_layers_ == config.num_hidden_layers
            assert two_ == 2
            assert num_heads_ == config.num_attention_heads
            assert features_ == (config.hidden_size // config.num_attention_heads)
            caches = tf.unstack(cache, axis=1)

        with tf.variable_scope(scope, default_name=self.config.scope_prefix, reuse=reuse):
            with tf.variable_scope("embeddings"):
                embeddings, self.embedding_table = embed(self.input_ids, config.vocab_size,
                                                         config.hidden_size,
                                                         position_offset=self.cache_length,
                                                         initializer_range=config.initializer_range,
                                                         max_position_embeddings=config.max_position_embeddings,
                                                         use_one_hot_embeddings=True)

            mask = get_attention_mask(self.seq_length, self.seq_length + self.cache_length, dtype=embeddings.dtype)

            # We keep the representation as a 2D tensor to avoid re-shaping it back and
            # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
            # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
            # help the optimizer.
            hidden_state = tf.reshape(embeddings, [self.batch_size * self.seq_length, self.config.hidden_size])
            new_kvs = []
            for layer_idx, layer_cache in enumerate(caches):
                with tf.variable_scope('layer{:02d}'.format(layer_idx)):
                    # [batch_size * seq_length, hidden_size]
                    attention_output, new_kv = attention_layer(
                        hidden_state,
                        mask,
                        batch_size=self.batch_size,
                        seq_length=self.seq_length,
                        size_per_head=config.hidden_size // config.num_attention_heads,
                        num_attention_heads=config.num_attention_heads,
                        initializer_range=config.initializer_range,
                        hidden_dropout_prob=self.config.hidden_dropout_prob,
                        attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
                        do_cache=do_cache,
                        cache=layer_cache,
                    )
                    new_kvs.append(new_kv)

                    # [batch_size * seq_length, hidden_size]
                    hidden_state = residual_mlp_layer(hidden_state + attention_output,
                                                      intermediate_size=config.intermediate_size,
                                                      hidden_dropout_prob=self.config.hidden_dropout_prob)
            self.hidden_state = hidden_state

        self.new_kvs = tf.stack(new_kvs, axis=1) if do_cache else None

        # Note that the hidden state is still flat (batch_size*hidden_size)
        self.logits_flat = tf.matmul(self.hidden_state, self.embedding_table, transpose_b=True)

        # THE OUTPUT BIAS DOES NOT SPARK JOY
        # output_bias = tf.get_variable('output_bias', shape=[config.vocab_size], initializer=tf.zeros_initializer())
        # self.logits_flat = tf.nn.bias_add(self.logits_flat, output_bias)

    @property
    def log_probs(self):
        logprobs_flat = tf.nn.log_softmax(self.logits_flat, axis=-1)
        return tf.reshape(logprobs_flat, [self.batch_size, self.seq_length, -1])

    def per_seq_prob(self, ignore_ids):
        """
        this will only take `clean' tokens i.e. there can be no trailing non-padding symbols after EOS!
        :return:
        """
        target_ids_flat = tf.reshape(self.target_ids, [-1])
        label_weights = tf.cast(tf.not_equal(target_ids_flat, self.pad_token_id), dtype=self.logits_flat.dtype)
        one_hot_labels = tf.one_hot(target_ids_flat,
                                    depth=self.config.vocab_size,
                                    dtype=self.logits_flat.dtype)
        logits_flat = self.logits_flat - tf.cast(ignore_ids[None], tf.float32) * 1e10
        logprobs_flat = tf.nn.log_softmax(logits_flat, axis=-1)
        selected_and_masked = label_weights * tf.reduce_sum(logprobs_flat * one_hot_labels, axis=[-1])
        per_seq_sum = tf.reduce_sum(tf.reshape(selected_and_masked, (-1, self.seq_length)), axis=[-1])
        return per_seq_sum

    def lm_loss(self):
        """
        :return: stuff
        """
        target_ids_flat = tf.reshape(self.target_ids, [-1])

        # 1 if it's valid and 0 otherwise.
        label_weights = tf.cast(tf.not_equal(target_ids_flat, self.pad_token_id), dtype=self.logits_flat.dtype)

        # [batch_size * seq_length, vocab_size]
        one_hot_labels = tf.one_hot(target_ids_flat,
                                    depth=self.config.vocab_size,
                                    dtype=self.logits_flat.dtype)

        # [batch_size * seq_length, vocab_size]
        logprobs_flat = tf.nn.log_softmax(self.logits_flat, axis=-1)

        per_example_loss = -tf.reduce_sum(logprobs_flat * one_hot_labels, axis=[-1])

        # per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_flat, labels=target_ids_flat)

        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator
        return loss

    def pooled_output(self, clf_token):
        """
        Extract pooled output given a token that says where we should look
        :param clf_token:
        :return:
        """
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(self.input_ids, clf_token), tf.float32), 1), tf.int32)
        return tf.gather(self.hidden_state, tf.range(self.batch_size, dtype=tf.int32) * self.seq_length + pool_idx)


def model_fn_builder(config: GroverConfig, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = GroverModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            pad_token_id=config.pad_token_id,
            chop_off_last_token=True,
        )

        total_loss = model.lm_loss()

        if is_training:
            train_op, train_metrics = optimization_adafactor.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        else:
            train_op = None
            train_metrics = {}
            tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            if use_tpu:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    host_call=construct_scalar_host_call(metric_dict=train_metrics, model_dir=params['model_dir'],
                                                         prefix='training/'),
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    training_hooks=[
                        tf.train.LoggingTensorHook({'loss': tf.metrics.mean(total_loss)[1]}, every_n_iter=100)],
                    scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(total_loss):
                loss = tf.metrics.mean(values=total_loss)
                return {
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [total_loss])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            gt_logprobs = tf.squeeze(tf.batch_gather(model.log_probs, model.target_ids[:, :, None]), axis=2)

            # Need top-p required under topp sampling!
            better_than_gt = model.log_probs > gt_logprobs[:, :, None]
            top_p_required = tf.reduce_sum(tf.cast(better_than_gt, tf.float32) * tf.exp(model.log_probs), axis=2)

            # No top-p sampling for now, since this seems to be too slow on TPUs
            if use_tpu:
                predictions = tf.reshape(
                    tf.random.categorical(logits=model.logits_flat, num_samples=1),
                    get_shape_list(model.target_ids),
                )
            else:
                # Argmax
                # predictions = tf.math.argmax(model.log_probs, axis=-1, output_type=tf.int32)
                predictions = tf.reshape(
                    _top_p_sample(model.logits_flat, num_samples=1, p=0.99)['sample'],
                    get_shape_list(model.target_ids),
                )
            pred_logprobs = tf.squeeze(tf.batch_gather(model.log_probs, predictions[:, :, None]), axis=2)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'gt_logprobs': gt_logprobs,
                             'top_p_required': top_p_required,
                             'predictions': predictions,
                             'pred_logprobs': pred_logprobs,
                             'labels': input_ids},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def nce_model_fn_builder(config: GroverConfig, init_checkpoint,
                         learning_rate,
                         num_train_steps, num_warmup_steps,
                         gen_checkpoint: Optional[str] = None,
                         correction_factor: float = 1., niter = 1,
                         ):

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        noises = features['noises']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        reuse_gen = config.reuse_gen
        residual_model = GroverModelResidual(
            scope='newslm',
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            noises=noises,
            pad_token_id=config.pad_token_id,
            reuse=tf.AUTO_REUSE,
            ignore_noise=(mode == tf.estimator.ModeKeys.PREDICT)
        )

        total_loss = residual_model.total_loss()
        reg_loss = 0

        if is_training:
            if config.regularize_g:
                reg_loss = residual_model.reg_g_loss()
                loss_to_optimize = total_loss + 1e-2 * reg_loss
            else:
                loss_to_optimize = total_loss
            train_op, train_metrics, acc_hook = optimization_adafactor.create_optimizer(
                loss_to_optimize, learning_rate, num_train_steps, num_warmup_steps, niter)
            tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        else:
            train_op = None
            tvars = tf.trainable_variables()
            acc_hook = None

        if reuse_gen:
            assert gen_checkpoint is not None
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, gen_checkpoint, only_initialize_prefix='newslm')
            tf.train.init_from_checkpoint(gen_checkpoint, assignment_map)

        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[
                    tf.train.LoggingTensorHook({'mean total loss': tf.metrics.mean(total_loss)[1],
                                                'mean reg loss': tf.metrics.mean(reg_loss)[1]}, every_n_iter=100),
                    acc_hook],
                )

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(total_loss):
                loss = tf.metrics.mean(values=total_loss)
                return {
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [total_loss])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                )
        elif mode == tf.estimator.ModeKeys.PREDICT:
            raise NotImplementedError
        else:
            raise NotImplementedError

        return output_spec

    return model_fn


def sample_step(tokens, ignore_ids,
                news_config, batch_size=1, p_for_topp=0.95, cache=None, do_topk=False,
                seed: Optional[int] = None, vanilla: bool = False):
    """
    Helper function that samples from grover for a single step
    :param tokens: [batch_size, n_ctx_b] tokens that we will predict from
    :param ignore_ids: [n_vocab] mask of the tokens we don't want to predict
    :param news_config: config for the GroverModel
    :param batch_size: batch size to use
    :param p_for_topp: top-p or top-k threshold
    :param cache: [batch_size, news_config.num_hidden_layers, 2,
                   news_config.num_attention_heads, n_ctx_a,
                   news_config.hidden_size // news_config.num_attention_heads] OR, None
    :return: new_tokens, size [batch_size]
             new_probs, also size [batch_size]
             new_cache, size [batch_size, news_config.num_hidden_layers, 2, n_ctx_b,
                   news_config.num_attention_heads, news_config.hidden_size // news_config.num_attention_heads]
    """
    model = GroverModel(
        config=news_config,
        is_training=False,
        input_ids=tokens,
        reuse=tf.AUTO_REUSE,
        scope='newslm',
        chop_off_last_token=False,
        do_cache=True,
        cache=cache,
    )

    # Extract the FINAL SEQ LENGTH
    batch_size_times_seq_length, vocab_size = get_shape_list(model.logits_flat, expected_rank=2)
    next_logits = tf.reshape(model.logits_flat, [batch_size, -1, vocab_size])[:, -1]

    if vanilla:
        sample_info = _top_p_sample(next_logits, ignore_ids=ignore_ids, num_samples=1, p=p_for_topp,
                                    seed=seed, vanilla=True)
    else:
        if do_topk:
            sample_info = _top_k_sample(next_logits, num_samples=1, k=tf.cast(p_for_topp, dtype=tf.int32))
        else:
            sample_info = _top_p_sample(next_logits, ignore_ids=ignore_ids, num_samples=1, p=p_for_topp,
                                        seed=seed, vanilla=False)

    new_tokens = tf.squeeze(sample_info['sample'], 1)
    new_probs = tf.squeeze(tf.batch_gather(sample_info['probs'], sample_info['sample']), 1)
    return {
        'new_tokens': new_tokens,
        'new_probs': new_probs,
        'new_cache': model.new_kvs,
    }


def initialize_from_context(initial_context, ignore_ids, news_config, p_for_topp=0.95, do_topk=False,
                            vanilla: bool = False):
    """ same signature as sample_step"""
    batch_size, _ = get_shape_list(initial_context, expected_rank=2)

    context_output = sample_step(tokens=initial_context, ignore_ids=ignore_ids, news_config=news_config,
                                 batch_size=batch_size, p_for_topp=p_for_topp, cache=None, do_topk=do_topk,
                                 vanilla=vanilla)
    return {
        'tokens': tf.concat([initial_context, context_output['new_tokens'][:, None]], 1),
        'cache': context_output['new_cache'],
        'probs': context_output['new_probs'][:, None]
    }


def eval_seq(news_config: GroverConfig, tokens, correction_factor = 1., baseline: bool = False, gen_scope='gen',
             ignore_ids: Optional = None):
    with tf.name_scope('evaluate_sequence'):
        gen_model = GroverModel(
            config=news_config,
            is_training=False,
            input_ids=tokens,
            reuse=tf.AUTO_REUSE,
            scope=gen_scope,
            chop_off_last_token=True,
            do_cache=True,
            cache=None,
        )
        lm_score = gen_model.per_seq_prob(ignore_ids=ignore_ids)
        if baseline:
            return lm_score

        residual_model = GroverModelResidual(
            config=news_config,
            is_training=False,
            input_ids=tokens,
            reuse=tf.AUTO_REUSE,
            noises=None,
            pad_token_id=news_config.pad_token_id,
            ignore_noise=True,
            scope='dis', ignore_ids=ignore_ids
        )
        residuals = residual_model.residuals[:, 0]
        unnormalized_p = residuals + correction_factor * lm_score
        return unnormalized_p


def sample(news_config: GroverConfig, initial_context, eos_token, ignore_ids=None, p_for_topp=0.95,
           do_topk=False, seed: Optional[int] = None, max_out_tensor: bool = False,
           vanilla: bool = False):
    """
    V1 version of: sample outputs from a model, and do it all at once
    :param news_config: Configuration used to construct the model
    :param initial_context: [batch_size, seq_length] that we'll start generating with
    :param eos_token: Stop generating if you see this (tf scalar)
    :param ignore_ids: NEVER GENERATE THESE [vocab_size]
    :return:
    """
    batch_size, _ = get_shape_list(initial_context, expected_rank=2)

    if ignore_ids is None:
        ignore_ids = tf.constant([x == 0 for x in range(news_config.vocab_size)], dtype=tf.bool)

    with tf.name_scope('sample_sequence'):
        # Initial call to get cache
        context_output = initialize_from_context(initial_context, ignore_ids=ignore_ids, news_config=news_config,
                                                 p_for_topp=p_for_topp,
                                                 do_topk=do_topk, vanilla=vanilla)
        ctx = context_output['tokens']
        cache = context_output['cache']
        probs = context_output['probs']

        def body(ctx, cache, probs):
            """ for whatever reason this didn't work when I ran it on more than one at once... ugh."""
            next_outputs = sample_step(ctx[:, -1][:, None], ignore_ids=ignore_ids, news_config=news_config,
                                       batch_size=batch_size, p_for_topp=p_for_topp, cache=cache,
                                       do_topk=do_topk,
                                       seed=seed, vanilla=vanilla)

            # Update everything
            new_cache = tf.concat([cache, next_outputs['new_cache']], axis=-2)
            new_ids = tf.concat([ctx, next_outputs['new_tokens'][:, None]], axis=1)
            new_probs = tf.concat([probs, next_outputs['new_probs'][:, None]], axis=1)
            return [new_ids, new_cache, new_probs]

        def cond(ctx, cache, probs):
            if max_out_tensor:
                return True
            is_eos = tf.equal(ctx, eos_token)
            return tf.math.logical_not(tf.reduce_all(tf.reduce_any(is_eos, axis=1)))

        tokens, cache, probs = tf.while_loop(
            cond=cond, body=body, maximum_iterations=1025 - get_shape_list(ctx)[1],
            loop_vars=[ctx, cache, probs],
            shape_invariants=[tf.TensorShape([batch_size, None]),
                              tf.TensorShape(
                                  [batch_size, news_config.num_hidden_layers, 2,
                                   news_config.num_attention_heads,
                                   None, news_config.hidden_size // news_config.num_attention_heads]),
                              tf.TensorShape([batch_size, None]),
                              ],
            back_prop=False,
        )

    return tokens, probs


def classification_model_fn_builder(config: GroverConfig, init_checkpoint, learning_rate,
                                    num_train_steps, num_warmup_steps, use_tpu, num_labels, pool_token_id,
                                    adafactor=False, adam_bfloat=False, lm_loss_coef=0.5):
    """Returns `model_fn` closure for TPUEstimator. FOR CLASSIFICATION ONLY!"""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        label_ids = features["label_ids"]
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # Create model with aux loss
        model = GroverModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            pad_token_id=config.pad_token_id,
            chop_off_last_token=False,
        )

        with tf.variable_scope('classification'):
            hidden_state = model.pooled_output(pool_token_id)
            if is_training:
                hidden_state = dropout(hidden_state, dropout_prob=0.1)
            logits = tf.layers.dense(
                hidden_state,
                num_labels,
                kernel_initializer=create_initializer(config.initializer_range),
                name='logits'
            )
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            class_loss = tf.reduce_mean(per_example_loss)

        total_loss = lm_loss_coef * model.lm_loss() + class_loss

        if is_training:
            train_op, train_metrics = optimization_adafactor.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            # tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            tvars = tf.trainable_variables()

            train_metrics['minibatch_cls_loss'] = class_loss
            train_metrics['minibatch_acc'] = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(logits, axis=-1, output_type=tf.int32),
                                 label_ids), tf.float32))
        else:
            train_op = None
            train_metrics = {}
            tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            if use_tpu:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    host_call=construct_scalar_host_call(metric_dict=train_metrics, model_dir=params['model_dir'],
                                                         prefix='training/'),
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    training_hooks=[
                        tf.train.LoggingTensorHook({'loss': tf.metrics.mean(total_loss)[1]}, every_n_iter=100)],
                    scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={'logits': logits,
                             'probs': tf.nn.softmax(logits, axis=-1)},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn
