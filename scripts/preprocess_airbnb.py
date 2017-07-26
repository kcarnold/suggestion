#!/usr/bin/env python

"""
Preprocess Airbnb data.
"""

__author__ = "Kenneth C. Arnold <kcarnold@alum.mit.edu>"


import os
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
import tqdm
import cytoolz
import operator
import nltk
from suggestion import tokenization

def flatten_dict(x, prefix=''):
    result = {}
    for k, v in x.items():
        if isinstance(v, dict):
            result.update(flatten_dict(v, prefix=k+'_'))
        else:
            result[prefix + k] = v
    return result

def tokenize(text):
    text = tokenization.URL_RE.sub(" ", text)
    sents = nltk.sent_tokenize(text)
    token_spaced_sents = (' '.join(sent[a:b] for a, b in tokenization.token_spans(sent)) for sent in sents)
    return '\n'.join(token_spaced_sents)

def build_vocab(tokenized_texts, min_occur_count):
    word_counts = cytoolz.frequencies(w for doc in tokenized_texts for w in doc.lower().split())
    word_counts = cytoolz.valfilter(lambda v: v >= min_occur_count, word_counts)
    vocab, counts = zip(*sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True))
    vocab = list(vocab)
    counts = np.array(counts)
    return vocab, counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path',
                        help='Path to Airbnb dataset',
                        default='~/Data/Airbnb')
    parser.add_argument('--outdir', help='Output directory',
                        default='airbnb_preproc')
    parser.add_argument('--valid-frac', type=float,
                        default=.05,
                        help="Fraction of data to use for validation")
    parser.add_argument('--test-frac', type=float,
                        default=.05,
                        help="Fraction of data to use for testing")
    parser.add_argument('--min-word-count', type=int,
                        default=10,
                        help="Minimum number of times a word can occur")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor()

    print("Loading Airbnb...", flush=True)
    raw_listings = pd.concat([pd.read_csv(x) for x in glob.glob('/Data/Airbnb/*/listings*')], axis=0)
    data = raw_listings[raw_listings.space.str.len() > 10].copy()
    del raw_listings
    data['tokenized'] = list(tqdm.tqdm(pool.map(tokenize, data['space'], chunksize=128), desc="Tokenizing", total=len(data), smoothing=0))
    data = data[data.tokenized.str.len() > 0]

    print("Splitting into train, validation, and test...", flush=True)
    train_frac = 1 - args.valid_frac - args.test_frac
    num_docs = len(data)
    indices = np.random.permutation(num_docs)
    splits = (np.cumsum([train_frac, args.valid_frac]) * num_docs).astype(int)
    segment_indices = np.split(indices, splits)
    names = ['train', 'valid', 'test']
    print(', '.join('{}: {}'.format(name, len(indices))
        for name, indices in zip(names, segment_indices)))
    train_indices = segment_indices[0]

    all_vocab = build_vocab(data['tokenized'], min_occur_count=2)
    pickle.dump(dict(
        vocab=all_vocab, data=data), open(os.path.join(args.outdir, 'all_data.pkl'), 'wb'), -1)

    train_vocab = build_vocab(
        data['tokenized'].iloc[train_indices],
        args.min_word_count)

    for name, indices in zip(names, segment_indices):
        cur_data = dict(vocab=train_vocab, data=data.iloc[indices].reset_index(drop=True))
        pickle.dump(cur_data, open(os.path.join(args.outdir, '{}_data.pkl'.format(name)), 'wb'), -1)
