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
    return probs

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp')
    parser.add_argument('--noises')

    args = parser.parse_args()

    probs = compute_nce_probs(args.inp, args.noises)
    print(probs)
    greater_than_chance = probs > - np.log(probs.shape[1])
    print(greater_than_chance)
    print(np.sum(greater_than_chance))

if __name__ == '__main__':
    main()