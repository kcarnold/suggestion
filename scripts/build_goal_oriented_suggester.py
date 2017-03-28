from suggestion import clustering
from scipy.special import entr

import tqdm
import logging
import pandas as pd
import pickle
import numpy as np
import cytoolz

logging.basicConfig(level=logging.INFO)

logging.info("Load all the reviews.")
data = pickle.load(open('yelp_preproc/all_data.pkl','rb'))
vocab, counts = data['vocab']
reviews = data['data'].reset_index(drop=True)
del data

freqs = counts / counts.sum()
word2idx = {word: idx for idx, word in enumerate(vocab)}
log_freqs = np.log(freqs)

logging.info("Compute unigram log likelihoods at the sentence level.")
def lookup_indices(sent):
    tmp = (word2idx.get(word) for word in sent)
    return [w for w in tmp if w is not None]

def mean_log_freq(indices):
    return np.mean(log_freqs[indices]) if len(indices) else None

def min_log_freq(indices):
    return np.min(log_freqs[indices]) if len(indices) else None

doc_sentence_indices = [[lookup_indices(sent.split()) for sent in doc.split('\n')] for doc in reviews.tokenized]
mean_llk = [list(cytoolz.filter(None, [mean_log_freq(indices) for indices in doc_indices])) for doc_indices in doc_sentence_indices]
min_llk = [list(cytoolz.filter(None, [min_log_freq(indices) for indices in doc_indices])) for doc_indices in doc_sentence_indices]
mean_mean_llk = pd.Series([np.mean(llks) if len(llks) > 0 else None for llks in mean_llk])
mean_min_llk = pd.Series([np.mean(llks) if len(llks) > 0 else None for llks in min_llk])

logging.info("Identify the best reviews.")
# Mark the top reviews: top-5 ranked reviews of restaurants with at least the median # reviews,
# as long as they have >= 10 votes.
reviews['total_votes'] = reviews['votes_cool'] + reviews['votes_funny'] + reviews['votes_useful']
reviews['total_votes_rank'] = reviews.groupby('business_id').total_votes.rank(ascending=False)
business_review_counts = reviews.groupby('business_id').review_count.mean()
median_review_count = np.median(business_review_counts)
yelp_is_best = (reviews.review_count >= median_review_count) & (reviews.total_votes >= 10) & (reviews.total_votes_rank <= 5)

#%%
num_sents = np.array([len(text.split('\n')) for text in reviews.tokenized])

logging.info("Build clusterizer")
clizer = clustering.Clusterizer.build(n_clusters=10)

logging.info("Analyze cluster distribution in all docs")

def normal_lik(x, sigma):
    return np.exp(-.5*(x/sigma)**2) / (2*np.pi*sigma)

def normalize_dists(dists):
    return dists / (np.sum(dists, axis=len(dists.shape)-1, keepdims=True) + 1e-6)

def cluster_dist_in_doc(doc_tokenized):
    vecs = clizer.vectorize_sents(doc_tokenized.split('\n'))
    norms = np.linalg.norm(vecs, axis=1)
    vecs = vecs[norms > .5]
    if len(vecs) == 0:
        return np.zeros(clizer.n_clusters)
    return np.mean(normalize_dists(normal_lik(clizer.clusterer.transform(vecs), .5)), axis=0)

cluster_dist_in_all_docs = np.array([cluster_dist_in_doc(tokenized) for tokenized in tqdm.tqdm(reviews.tokenized)])
cluster_dist_in_all_docs_norm = normalize_dists(cluster_dist_in_all_docs)

target_dists = dict(
     overall=normalize_dists(np.mean(cluster_dist_in_all_docs_norm, axis=0)),
     best=normalize_dists(np.mean(cluster_dist_in_all_docs_norm[yelp_is_best], axis=0)),
     rest=normalize_dists(np.mean(cluster_dist_in_all_docs_norm[~yelp_is_best], axis=0)))
# TODO: write this somewhere.
print({k: entr(v).sum() for k, v in target_dists.items()})

#%%
entropies = np.sum(entr(normalize_dists(cluster_dist_in_all_docs)), axis=1)
#%%
def plot_cluster_entropies():
    long_enough = pd.Series(num_sents > 2)# & pd.Series(num_sents < 10)
    bw=.04
    to_plot = pd.Series(entropies).dropna()
    to_plot = to_plot[to_plot > 0]
    clip = np.percentile(to_plot, [2.5, 97.5])
    sns.kdeplot(to_plot[yelp_is_best & long_enough], clip=clip, label=f'Yelp best (mean={to_plot[yelp_is_best & long_enough].mean():.2f})', bw=bw)
    sns.kdeplot(to_plot[~yelp_is_best & long_enough].dropna(), clip=clip, label=f'Yelp rest (mean={to_plot[~yelp_is_best & long_enough].mean():.2f})', bw=bw)
    plt.xlabel("Entropy of cluster distribution")
    plt.savefig('figures/cluster_distribution_entropy.pdf')
#%%
import re
cant_type = re.compile(r'[^\-a-z., !\']')
clizer.omit = [idx for idx, phrase in enumerate(clizer.unique_starts) if cant_type.search(' '.join(phrase))]
clizer.target_dists = target_dists
#%%
import os
from suggestion.paths import paths
with open(os.path.join(paths.parent, 'models', 'goal_oriented_suggestion_data.pkl'), 'wb') as f:
    pickle.dump(clizer, f, -1)

##%%
#import numpy as np
#existing_dists = np.zeros((0, clizer.n_clusters))
##existing_dists[0,2] = .5
#new_dists_opts = np.eye(clizer.n_clusters)
#with_new_dist = np.array([np.concatenate((existing_dists, new_dist_opt[None]), axis=0) for new_dist_opt in new_dists_opts])
#dist_with_new_dist = normalize_dists(np.mean(with_new_dist, axis=1))
#from scipy.special import kl_div
#np.argsort(kl_div(dist_with_new_dist, target_dists['best']).sum(axis=1))[:3]
##def kl_divergence_after_addition(existing_dists, new_dist)
