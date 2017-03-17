# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:32:52 2017

@author: kcarnold
"""
import pandas as pd
import pickle
import numpy as np
#%% Load all the reviews.
data = pickle.load(open('yelp_preproc/all_data.pkl','rb'))
vocab, freqs = data['vocab']
reviews = data['data'].reset_index(drop=True)
del data
#%%
word2idx = {word: idx for idx, word in enumerate(vocab)}
log_freqs = np.log(freqs)
#%%
def lookup_indices(sent):
    tmp = (word2idx.get(word) for word in sent)
    return [w for w in tmp if w is not None]

def mean_log_freq(indices):
    return np.mean(log_freqs[indices]) if len(indices) else None

def min_log_freq(indices):
    return np.min(log_freqs[indices]) if len(indices) else None

doc_sentence_indices = [[lookup_indices(sent.split()) for sent in doc.split('\n')] for doc in reviews.tokenized]
#%%
import cytoolz
mean_llk = [list(cytoolz.filter(None, [mean_log_freq(indices) for indices in doc_indices])) for doc_indices in doc_sentence_indices]
min_llk = [list(cytoolz.filter(None, [min_log_freq(indices) for indices in doc_indices])) for doc_indices in doc_sentence_indices]
#%%
mean_mean_llk = pd.Series([np.mean(llks) if len(llks) > 0 else None for llks in mean_llk])
mean_min_llk = pd.Series([np.mean(llks) if len(llks) > 0 else None for llks in min_llk])

#%% Identify the best reviews.
# Mark the top reviews: top-5 ranked reviews of restaurants with at least the median # reviews,
# as long as they have >= 10 votes.
reviews['total_votes'] = reviews['votes_cool'] + reviews['votes_funny'] + reviews['votes_useful']
reviews['total_votes_rank'] = reviews.groupby('business_id').total_votes.rank(ascending=False)
business_review_counts = reviews.groupby('business_id').review_count.mean()
median_review_count = np.median(business_review_counts)
yelp_is_best = (reviews.review_count >= median_review_count) & (reviews.total_votes >= 10) & (reviews.total_votes_rank <= 5)

#%%
import seaborn as sns
import matplotlib.pyplot as plt
to_plot = mean_min_llk.dropna()
clip = np.percentile(to_plot, [2.5, 97.5])
sns.kdeplot(to_plot[yelp_is_best], clip=clip, label='Yelp best')
sns.kdeplot(to_plot[~yelp_is_best].dropna(), clip=clip, label='Yelp rest')
