#! /usr/bin/env python3


import argparse
from pathlib import Path

from pytorch_lightning import seed_everything

from qa_utils.datasets.antique import ANTIQUE
from qa_utils.datasets.fiqa import FiQA
from qa_utils.datasets.insuranceqa import InsuranceQA
from qa_utils.datasets.msmarco import MSMARCO


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('SAVE', help='Where to save the results')
    ap.add_argument('--num_negatives', type=int, default=1, help='Number of negatives per positive (pointwise training)')
    ap.add_argument('--pw_num_negatives', type=int, default=16, help='Number of negatives per positive (pairwise training)')
    ap.add_argument('--pw_query_limit', type=int, default=64, help='Maximum number of training examples per query (pairwise training)')
    ap.add_argument('--random_seed', type=int, default=123, help='Random seed')

    subparsers = ap.add_subparsers(help='Choose a dataset', dest='dataset')
    subparsers.required = True
    for c in [ANTIQUE, FiQA, InsuranceQA, MSMARCO]:
        c.add_subparser(subparsers, c.__name__.lower())
    args = ap.parse_args()

    if args.random_seed:
        seed_everything(args.random_seed)

    if args.dataset == FiQA.__name__.lower():
        ds = FiQA(args)
    elif args.dataset == InsuranceQA.__name__.lower():
        ds = InsuranceQA(args)
    elif args.dataset == MSMARCO.__name__.lower():
        ds = MSMARCO(args)
    elif args.dataset == ANTIQUE.__name__.lower():
        ds = ANTIQUE(args)
    else:
        raise argparse.ArgumentError('Unsupported dataset')

    save_path = Path(args.SAVE)
    save_path.mkdir(parents=True, exist_ok=True)
    ds.save(save_path, args.num_negatives, args.pw_num_negatives, args.pw_query_limit)


if __name__ == '__main__':
    main()
