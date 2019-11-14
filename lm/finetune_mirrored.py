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

from lm.dataloader import input_fn_builder
from lm.modeling import model_fn_builder, GroverConfig

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

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for adafactor.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer("seed", 42, "Random seed.")

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


def main(_):
    from random import seed as rnd_seed
    rnd_seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    import os
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    strategy = \
        tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.NCCL)

    tf.logging.set_verbosity(tf.logging.INFO)

    news_config = GroverConfig.from_json_file(FLAGS.config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    # tf.logging.info("*** Input Files ***")
    # for input_file in input_files:
    #     tf.logging.info("  %s" % input_file)

    run_config = tf.ConfigProto(allow_soft_placement=True)
    run_config.gpu_options.allow_growth = True
    model_dir = FLAGS.output_dir

    model_fn = model_fn_builder(news_config, init_checkpoint=FLAGS.init_checkpoint,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=FLAGS.num_train_steps,
                                num_warmup_steps=FLAGS.num_warmup_steps,
                                use_tpu=FLAGS.use_tpu,
                                )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    est = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            session_config=run_config,
            train_distribute=strategy,
            tf_random_seed=FLAGS.seed,
            keep_checkpoint_max=0,),
        model_dir=model_dir,
        params={'model_dir': model_dir}
    )

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        seq_length=FLAGS.max_seq_length,
        is_training=True)

    eval_input_fn = input_fn_builder(
        input_files=[],
        seq_length=FLAGS.max_seq_length,
        is_training=True)

    tf.estimator.train_and_evaluate(est,
                                    train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps),
                                    eval_spec=tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=2000),
                                    )


def set_tf_config():
    import os
    import json
    from hostlist import expand_hostlist

    host_list = expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

    start_port = 12345

    local_name = os.environ['SLURMD_NODENAME']
    # FIXME this is brittle because we may not have same # of gpus per machine
    # FIXME however tf only supports same configuration over all machines now, anyway
    num_workers_per_machine = max(int(os.environ['SLURM_NTASKS']) // len(host_list), 1)
    local_id = int(os.environ['SLURM_LOCALID'])
    rank = int(os.environ['SLURM_PROCID'])
    tf_config_json = {
        'cluster': {
            'worker': []
        },
        'task': {'type': 'worker', 'index': -1}
    }

    for host_idx, host in enumerate(host_list):
        for task_id in range(num_workers_per_machine):
            tf_config_json['cluster']['worker'].append('{}:{}'.format(host, start_port+task_id))
            if host == local_name and task_id == local_id:
                tf_config_json['task']['index'] = (num_workers_per_machine * host_idx) + local_id
    assert tf_config_json['task']['index'] != -1, '{} {} {}'.format(tf_config_json, local_name, local_id)
    print(tf_config_json)
    os.environ['TF_CONFIG'] = json.dumps(tf_config_json)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    set_tf_config()
    tf.app.run()
