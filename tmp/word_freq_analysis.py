# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:32:52 2017

@author: kcarnold
"""
import json
import pickle
import numpy as np
#%% Load all the reviews.
restaurant_reviews = json.load(open('models/reviews.json'))
tokenized_reviews = pickle.load(open('models/tokenized_reviews.pkl','rb'))
#%%
assert len(restaurant_reviews) == len(tokenized_reviews)
#%%
#from collections import Counter
import cytoolz
word_counts = cytoolz.frequencies(cytoolz.concat(tokenized_reviews))
#%%
normal_word_counts = cytoolz.keyfilter(lambda k: k[0] not in '<,!.?\'-"', word_counts)
normal_word_counts = cytoolz.valfilter(lambda v: v > 1, normal_word_counts)
#%%
import operator
vocab, counts = zip(*sorted(normal_word_counts.items(), key=operator.itemgetter(1), reverse=True))
vocab = list(vocab)
counts = np.array(counts)
freqs = counts / counts.sum()
word2idx = {word: idx for idx, word in enumerate(vocab)}
log_freqs = np.log(freqs)
#%%
def split_sents(tokenized):
    return [x.split() for x in ' '.join(tokenized).replace('<D> ', '').replace('<P> ', '').replace('<S> ','').split('</S>') if x]
doc_sentences = [split_sents(r) for r in tokenized_reviews]
#%%
def lookup_indices(sent):
    tmp = (word2idx.get(word) for word in sent)
    return [w for w in tmp if w is not None]

def mean_log_freq(indices):
    return np.mean(log_freqs[indices]) if len(indices) else None

def min_log_freq(indices):
    return np.min(log_freqs[indices]) if len(indices) else None

doc_sentence_indices = [[lookup_indices(sent) for sent in doc] for doc in doc_sentences]
#%%
mean_llk = [[mean_log_freq(indices) for indices in doc_indices] for doc_indices in doc_sentence_indices]
min_llk = [[min_log_freq(indices) for indices in doc_indices] for doc_indices in doc_sentence_indices]
#%%
# Mark the top reviews: top-5 ranked reviews of restaurants with at least the median # reviews,
# as long as they have >= 10 votes.

yelp_reviews['total_votes'] = yelp_reviews['votes_cool'] + yelp_reviews['votes_funny'] + yelp_reviews['votes_useful']
yelp_reviews['total_votes_rank'] = yelp_reviews.groupby('business_id').total_votes.rank(ascending=False)
business_review_counts = yelp_reviews.groupby('business_id').review_count.mean()
median_review_count = np.median(business_review_counts)
yelp_is_best = (yelp_reviews.review_count >= median_review_count) & (yelp_reviews.total_votes >= 10) & (yelp_reviews.total_votes_rank <= 5)
