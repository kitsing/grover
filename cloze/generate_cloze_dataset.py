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
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--choice', default=0, type=int)
    parser.add_argument('--output-path', default='./')

    return parser.parse_args()


def main():
    args = parse_args()
    if 'SLURM_PROCID' in environ:
        args.seed = args.seed + int(environ['SLURM_PROCID'])
    random.seed(args.seed)
    encoder = get_encoder()

    example = tf.train.Example()
    position = None
    record = None
    answer = None
    for record_idx, record in enumerate(tf.python_io.tf_record_iterator(args.file)):
        if record_idx < args.choice:
            continue
        if record_idx > args.choice:
            break
        example.ParseFromString(record)
        record = example.features.feature['input_ids'].int64_list.value
        parsed_length = sum([1 for _ in record if _ != args.pad])
        position = random.randint(1, parsed_length - 2)
        answer = record[position]

    expanded = np.array([record,] * len(encoder), dtype=np.int)
    expanded[:, position] = np.arange(len(encoder))

    out_fname = f'{args.output_path}/{basename(args.file)}.npz'
    np.savez(file=out_fname, answer=answer, position=position, cloze=expanded)

if __name__ == '__main__':
    main()