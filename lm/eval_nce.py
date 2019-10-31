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

""" Training script! """
import tensorflow as tf

from lm.dataloader import nce_input_fn_builder
from lm.modeling import nce_model_fn_builder, GroverConfig

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "config_file", 'configs/base.json',
    "The config json file corresponding to the pre-trained news model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "input_dev_file", None,
    "Input dev TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "noise_file", None,
    "Input noise files (can be a glob or comma separated).")

flags.DEFINE_string(
    "noise_dev_file", None,
    "Input dev noise files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained model).")

flags.DEFINE_integer(
    "max_seq_length", 1024,
    "The maximum total input sequence length after BPE tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

# flags.DEFINE_integer("train_batch_size", 2, "Total batch size for training.")
flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for adafactor.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_integer("seed", 42, "Random seed.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "eval_delay_secs", 300,
    "delay evaluation.")

flags.DEFINE_integer(
    "eval_throttle_secs", 1800,
    "delay evaluation.")


def main(_):
    from random import seed
    seed(FLAGS.seed)
    import os
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    tf.logging.set_verbosity(tf.logging.WARN)

    news_config = GroverConfig.from_json_file(FLAGS.config_file)
    print(news_config)
    if rank == 0:
        tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    input_dev_files = []
    for input_pattern in FLAGS.input_dev_file.split(","):
        input_dev_files.extend(tf.gfile.Glob(input_pattern))

    noise_files = []
    for noise_pattern in FLAGS.noise_file.split(","):
        noise_files.extend(tf.gfile.Glob(noise_pattern))

    noise_dev_files = []
    for noise_pattern in FLAGS.noise_dev_file.split(","):
        noise_dev_files.extend(tf.gfile.Glob(noise_pattern))

    # tf.logging.info("*** Input Files ***")
    # for input_file in input_files:
    #     tf.logging.info("  %s" % input_file)

    run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    model_dir = FLAGS.output_dir

    model_fn = nce_model_fn_builder(news_config, init_checkpoint=FLAGS.init_checkpoint,
                                    learning_rate=FLAGS.learning_rate,
                                    num_train_steps=FLAGS.num_train_steps,
                                    num_warmup_steps=FLAGS.num_warmup_steps,
                                    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    est = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(session_config=run_config,
            train_distribute=strategy,
            tf_random_seed=FLAGS.seed),
        model_dir=model_dir,
        params={'model_dir': model_dir}
    )

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

    eval_input_fn = nce_input_fn_builder(k=1, constant_noise=True,
                                         input_files=input_dev_files,
                                         noise_files=noise_dev_files,
                                         seq_length=FLAGS.max_seq_length,
                                         is_training=False,
                                         input_batch_size=FLAGS.train_batch_size)

    predicted = est.predict(input_fn=eval_input_fn, predict_keys=None,
                            start_delay_secs=FLAGS.eval_delay_secs,
                            )


def set_tf_config():
    import os
    import json
    from hostlist import expand_hostlist

    host_list = expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

    start_port = 12345

    rank = int(os.environ['SLURM_PROCID'])
    tf_config_json = {
        'cluster': {
            'worker': []
        },
        'task': {'type': 'worker', 'index': rank}
    }
    for host_idx, host in enumerate(host_list):
        tf_config_json['cluster']['worker'].append('{}:{}'.format(host, start_port))
    print(tf_config_json)
    os.environ['TF_CONFIG'] = json.dumps(tf_config_json)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    set_tf_config()
    tf.app.run()
