#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
from scipy.special import logsumexp
from os.path import basename

def output_loss(fname, ignored: np.ndarray, prob_path):
    prob_file = f'{prob_path}/{basename(fname)}.out.npz'
    with np.load(fname) as fh, np.load(prob_file) as prob_fh:
        prob_outputs = prob_fh['unnormalized_probs']
        assert prob_outputs.shape == ignored.shape
        prob_outputs = prob_outputs + ignored
        answer = fh['answer'][0]
        normalized_log_prob = prob_outputs[answer] - logsumexp(prob_outputs)
        with open(prob_file+'.answer') as answer_fh:
            answer_fh.write('{}\n'.format(normalized_log_prob))


def main():
    import argparse
    from glob import glob
    from sample.encoder import get_encoder

    encoder = get_encoder()
    ignore_ids_np = np.array(encoder.special_tokens_onehot, dtype=np.float)
    ignore_ids_np[encoder.__dict__['end_article']] = 0
    ignore_ids_np = np.log(ignore_ids_np)

    parser = argparse.ArgumentParser()
    parser.add_argument('--prob-path', default='./')
    parser.add_argument('--cloze-path', default='./*')

    args = parser.parse_args()
    cloze_files = glob(args.cloze_path)

    for f in tqdm(cloze_files):
        output_loss(f, ignored=ignore_ids_np, prob_path=args.prob_path)


if __name__ == '__main__':
    main()