#!/usr/bin/env python3
import numpy as np
from scipy.special import logsumexp

def compute_nce_probs(inp, noises):
    loaded = np.load(inp)
    loaded_noise = np.load(noises)
    noise_weighted = loaded_noise['noise_probs_under_model'] - loaded_noise['noise_probs_under_noise']
    true_weighted = loaded['input_probs_under_model'] - loaded['input_probs_under_noise']
    z = np.concatenate(
        (true_weighted.reshape((-1, 1)), np.tile(noise_weighted.reshape((1, -1)), (true_weighted.shape[0], 1))), axis=1)
    probs = true_weighted - logsumexp(z, axis=1)
    num_noises = z.shape[1]
    return probs, num_noises

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp')
    parser.add_argument('--noises')

    args = parser.parse_args()

    probs, num_noises = compute_nce_probs(args.inp, args.noises)
    print(np.mean(probs))
    greater_than_chance = probs > - np.log(num_noises)
    print(np.sum(greater_than_chance))

if __name__ == '__main__':
    main()
