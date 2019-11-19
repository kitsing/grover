#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import random
from sample.encoder import get_encoder
from os.path import basename
from os import environ

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default='a')
    parser.add_argument('--eos', default=50266, type=int)
    parser.add_argument('--bos', default=50265, type=int)
    parser.add_argument('--pad', default=0, type=int)
    parser.add_argument('--output-path', default='./')
    return parser.parse_args()


def main():
    args = parse_args()
    example = tf.train.Example()

    to_write = []
    for record_idx, record in enumerate(tf.python_io.tf_record_iterator(args.file)):
        example.ParseFromString(record)
        record = [int(_) for _ in example.features.feature['input_ids'].int64_list.value]
        to_write.append(np.array(record, dtype=np.int32).reshape((1, -1)))
    concatenated = np.concatenate(to_write, axis=0)

    out_fname = f'{args.output_path}/{basename(args.file)}.npz'
    np.savez(file=out_fname, tokens=concatenated)

if __name__ == '__main__':
    main()
