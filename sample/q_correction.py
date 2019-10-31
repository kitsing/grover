#!/usr/bin/env python3

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--eos', default=50266, type=int)

    return parser.parse_args()


def get_count(fname, eos):
    import numpy as np
    with np.load(fname) as loaded:
        t = loaded['tokens']
        sampled = sum((t == eos).argmax(axis=1) > 0)
        return int(sampled), t.shape[0]


def main():
    from tqdm import tqdm
    args = parse_args()
    from glob import glob
    filenames = glob(args.path)

    total = 0.
    sampled = 0.
    for f in tqdm(filenames, disable=None):
        local_sampled, local_total = get_count(f, args.eos)
        sampled += local_sampled
        total += local_total

    print('{}/{}'.format(sampled, total))


if __name__ == '__main__':
    main()
