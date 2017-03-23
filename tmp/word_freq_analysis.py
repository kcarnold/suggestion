# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:32:52 2017

@author: kcarnold
"""
import pandas as pd
import pickle
import numpy as np
import cytoolz

#%% Load all the reviews.
data = pickle.load(open('yelp_preproc/all_data.pkl','rb'))
vocab, counts = data['vocab']
reviews = data['data'].reset_index(drop=True)
del data
#%%
freqs = counts / counts.sum()
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
num_sents = np.array([len(text.split('\n')) for text in reviews.tokenized])
#%%
import seaborn as sns
import matplotlib.pyplot as plt
#%%
to_plot = mean_min_llk.dropna()
clip = np.percentile(to_plot, [2.5, 97.5])
sns.kdeplot(to_plot[yelp_is_best], clip=clip, label='Yelp best')
sns.kdeplot(to_plot[~yelp_is_best].dropna(), clip=clip, label='Yelp rest')
plt.xlabel("Mean min unigram log likelihood")
plt.savefig('figures/mean_min_unigram_llk_2.pdf')

#%%
to_plot = mean_mean_llk.dropna()
clip = np.percentile(to_plot, [2.5, 97.5])
sns.kdeplot(to_plot[yelp_is_best], clip=clip, label='Yelp best')
sns.kdeplot(to_plot[~yelp_is_best].dropna(), clip=clip, label='Yelp rest')
plt.xlabel("Mean mean unigram log likelihood")
plt.savefig('figures/mean_mean_unigram_llk_2.pdf')

#%% Analyze topic distributions
from suggestion import clustering
clizer = clustering.Clusterizer()
#%%
def clusters_in_doc(doc_tokenized):
    vecs = clizer.vectorize_sents(doc_tokenized.split('\n'))
    norms = np.linalg.norm(vecs, axis=1)
    vecs = vecs[norms > .5]
    if len(vecs) == 0:
        return np.zeros(clizer.n_clusters)
    return np.bincount(clizer.clusterer.predict(vecs), minlength=clizer.n_clusters)

clusters_in_doc(reviews.tokenized.iloc[50])
#%%
import tqdm
clusters_in_all_docs = np.array([clusters_in_doc(tokenized) for tokenized in tqdm.tqdm(reviews.tokenized)])
#%%
cluster_probs = clusters_in_all_docs / (np.sum(clusters_in_all_docs, axis=1, keepdims=True) + 1e-9)
#%%

def normal_lik(x, sigma):
    return np.exp(-.5*(x/sigma)**2) / (2*np.pi*sigma)

def normalize_dists(dists):
    return dists / (np.sum(dists, axis=1, keepdims=True) + 1e-6)
#%%
def cluster_dist_in_doc(doc_tokenized):
    vecs = clizer.vectorize_sents(doc_tokenized.split('\n'))
    norms = np.linalg.norm(vecs, axis=1)
    vecs = vecs[norms > .5]
    if len(vecs) == 0:
        return np.zeros(clizer.n_clusters)
    return np.mean(normalize_dists(normal_lik(clizer.clusterer.transform(vecs), .5)), axis=0)

cluster_dist_in_all_docs = np.array([cluster_dist_in_doc(tokenized) for tokenized in tqdm.tqdm(reviews.tokenized)])
#%%


stats = dict(
     overall=np.mean(cluster_probs, axis=0),
     best=np.mean(cluster_probs[yelp_is_best], axis=0),
     rest=np.mean(cluster_probs[~yelp_is_best], axis=0))
{k: entr(v).sum() for k, v in stats.items()}
#%%
np.mean(np.sum(cluster_probs, axis=1))
#%%
from scipy.special import entr
entropies = np.sum(entr(cluster_probs), axis=1)
#%%
entropies2 = np.sum(entr(normalize_dists(cluster_dist_in_all_docs)), axis=1)
#%%
long_enough = pd.Series(num_sents > 2)# & pd.Series(num_sents < 10)
bw=.04
to_plot = pd.Series(entropies2).dropna()
to_plot = to_plot[to_plot > 0]
clip = np.percentile(to_plot, [2.5, 97.5])
sns.kdeplot(to_plot[yelp_is_best & long_enough], clip=clip, label=f'Yelp best (mean={to_plot[yelp_is_best & long_enough].mean():.2f})', bw=bw)
sns.kdeplot(to_plot[~yelp_is_best & long_enough].dropna(), clip=clip, label=f'Yelp rest (mean={to_plot[~yelp_is_best & long_enough].mean():.2f})', bw=bw)
plt.xlabel("Entropy of cluster distribution")
plt.savefig('figures/cluster_distribution_entropy.pdf')