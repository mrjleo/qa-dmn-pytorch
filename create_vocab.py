#! /usr/bin/env python3


import pickle
import argparse
from itertools import chain
from collections import Counter

import h5py
import nltk
from tqdm import tqdm
from torchtext.vocab import Vocab


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('DATA_FILE', help='File that holds the queries and documents')
    ap.add_argument('--max_size', type=int, help='Maximum vocabulary size')
    ap.add_argument('--cache', help='Torchtext cache')
    ap.add_argument('--vectors', default='glove.840B.300d', help='Pre-trained vectors')
    ap.add_argument('--out_file', default='vocab.pkl', help='Where to save the vocabulary')
    args = ap.parse_args()

    print(f'reading {args.DATA_FILE}...')
    with h5py.File(args.DATA_FILE, 'r') as fp:
        num_items = len(fp['queries']) + len(fp['docs'])
        ct = Counter()
        for s in tqdm(chain(fp['queries'], fp['docs']), total=num_items):
            ct.update(nltk.word_tokenize(s.lower()))
        vocab = Vocab(ct, args.max_size, vectors=args.vectors, vectors_cache=args.cache)

    print(f'writing {args.out_file}...')
    with open(args.out_file, 'wb') as fp:
        pickle.dump(vocab, fp)


if __name__ == '__main__':
    main()
