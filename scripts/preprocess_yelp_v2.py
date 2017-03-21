#!/usr/bin/env python

"""
Preprocess Yelp data.
"""

__author__ = "Kenneth C. Arnold <kcarnold@alum.mit.edu>"


import os
import sys
import argparse
import gzip
import ujson as json
import pickle
import numpy as np
import nltk
import pandas as pd
import tqdm
import cytoolz
import operator
import spacy

nlp = None

def flatten_dict(x, prefix=''):
    result = {}
    for k, v in x.items():
        if isinstance(v, dict):
            result.update(flatten_dict(v, prefix=k+'_'))
        else:
            result[prefix + k] = v
    return result

def load_yelp(path):
    data_types = {x: [] for x in ['review', 'business', 'user']}
    with gzip.open(os.path.expanduser(path), 'rb') as f:
        for line in f:
            rec = json.loads(line.decode('utf8'))
            rec = flatten_dict(rec)
            data_types[rec['type']].append(rec)
    return data_types

def join_yelp(data):
    reviews = pd.DataFrame(data['review']).drop(['type'], axis=1)
    businesses = pd.DataFrame(data['business']).drop(['type', 'photo_url', 'url', 'full_address', 'schools'], axis=1)
    users = pd.DataFrame(data['user']).drop(['type', 'name', 'url'], axis=1)

    restaurants = businesses[businesses.open & businesses.categories.apply(lambda x: 'Restaurants' in x)]
    restaurants = restaurants.drop(['open'], axis=1)

    result = pd.merge(
        reviews, restaurants,
        left_on='business_id', right_on='business_id', suffixes=('_review','_biz'))
    result = pd.merge(
        result, users,
        left_on='user_id', right_on='user_id', suffixes=('','_user'))

    result['date'] = pd.to_datetime(result.date)
    def to_months(time_delta):
        return time_delta.total_seconds() / 3600. / 24. / 30.
    result['age_months'] = (result.date.max() - result.date).apply(to_months)
    return result

def tokenize(text):
    return '\n'.join(' '.join(word.text for word in sent).lower() for sent in nlp(text).sents)

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
                        help='Path to Yelp dataset',
                        default='~/Data/Yelp/yelp_academic_dataset.json.gz')
    parser.add_argument('--outdir', help='Output directory',
                        default='yelp_preproc')
    parser.add_argument('--valid-frac', type=float,
                        default=.05,
                        help="Fraction of data to use for validation")
    parser.add_argument('--test-frac', type=float,
                        default=.05,
                        help="Fraction of data to use for testing")
    parser.add_argument('--min-word-count', type=int,
                        default=10,
                        help="Minimum number of times a word can occur")
    parser.add_argument('--spacy', help="SpaCy model name",
                        default="en_core_web_md")
#    parser.add_argument('--max-vocab-size', type=int,
#                        default=5000,
#                        help="Maximum number of words in vocab")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print("Loading SpaCy model", args.spacy, flush=True)
    nlp = spacy.load(args.spacy)

    # Consistent train-test splits.
    np.random.seed(0)

    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor()

    print("Loading Yelp...", flush=True)
    data = join_yelp(load_yelp(path=args.path))
    data['tokenized'] = list(tqdm.tqdm(pool.map(tokenize, data['text'], chunksize=1024), desc="Tokenizing", total=len(data)))
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
