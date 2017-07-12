# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:07:12 2017

@author: kcarnold
"""


import numpy as np
import pandas as pd
from suggestion import clustering
#%%
from suggestion.analyzers import load_reviews
reviews = load_reviews()
#%%
sents_0 = [sent for doc in reviews.tokenized for sent in doc.lower().split('\n')]
doc_idx_0 = np.array([doc_idx for doc_idx, doc in enumerate(reviews.tokenized) for sent in doc.lower().split('\n')])
#%%
sent_lens = np.array([len(sent.split()) for sent in sents_0])
min_sent_len, max_sent_len = np.percentile(sent_lens, [25, 75])
indices_1 = np.flatnonzero((min_sent_len <= sent_lens) & (sent_lens <= max_sent_len))
sents_1 = [sents_0[i] for i in indices_1]

#%%# Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5, max_df=.5, stop_words='english')
vecs_1 = vectorizer.fit_transform(sents_1)
# Note: The vectorizer normalized the vectors.

#%%
n_clusters = 10
random_state = 0
from sklearn.cluster import MiniBatchKMeans
import wordfreq
cnnb = clustering.ConceptNetNumberBatch.load()

#%%
sklearn_vocab = vectorizer.get_feature_names()
def get_or_zero(cnnb, item):
    try:
        return cnnb[item]
    except KeyError:
        return np.zeros(cnnb.ndim)
cnnb_vecs_for_sklearn_vocab = np.array([get_or_zero(cnnb, word) for word in sklearn_vocab])
wordfreqs_for_sklearn_vocab = [wordfreq.word_frequency(word, 'en', 'large', minimum=1e-9) for word in sklearn_vocab]
projection_mat = -np.log(wordfreqs_for_sklearn_vocab)[:,None] * cnnb_vecs_for_sklearn_vocab

#%%
dense_vecs_1 = vecs_1.dot(projection_mat)

#%% Normalize, and filter by norm.
min_norm = .5
norms_1 = np.linalg.norm(dense_vecs_1, axis=1)
indices_2 = np.flatnonzero(norms_1 > min_norm)
dense_vecs_2 = dense_vecs_1[indices_2] / norms_1[indices_2][:,None]
sents_2 = [sents_1[i] for i in indices_2]


#%%
if False:
    from sklearn.utils import gen_batches
    from sklearn.metrics.pairwise import linear_kernel
    import tqdm
    num_sims = 32
    block_size = 128
    density = np.empty((projected_vecs.shape[0], num_sims))
    for indices in tqdm.tqdm(gen_batches(len(projected_vecs), block_size), total=len(projected_vecs)//block_size+1):
        sims = linear_kernel(projected_vecs, projected_vecs[indices]).T
        density[indices] = np.sort(np.partition(sims, -num_sims, axis=-1)[:,-num_sims:])
    #    for idx, sim in zip(range(*indices.indices(projected_vecs.shape[0])), sims):
    #        density[idx] = np.sort(np.partition(sim, -num_sims)[-num_sims:])
#%%
n_clusters = 128
mbk_wordvecs = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=random_state)
mbk_wordvecs.fit(dense_vecs_2)
#%%
from sklearn.metrics import pairwise_distances
import seaborn as sns
sns.heatmap(pairwise_distances(mbk_wordvecs.cluster_centers_), cmap='RdBu_r')
#%%
#dists_to_centers = pairwise_distances(mbk_wordvecs.cluster_centers_, projected_vecs, n_jobs=-1)
dists_to_centers = mbk_wordvecs.transform(dense_vecs_2)
#%%
orig_indices_2 = indices_1[indices_2]
#%%
def show_cluster(idx):
    close_indices = np.argsort(dists_to_centers[:,idx])[:10]
    for idx in close_indices:
        print(sents_2[idx])
#%%
all_stars = []
for i in range(dists_to_centers.shape[1]):
    print()
    print(i)
    close_indices = np.argsort(dists_to_centers[:,i])[:10]
    for idx in close_indices:
        print(sents_2[idx])
    orig_docs = doc_idx_0[orig_indices_2[close_indices]]
    orig_stars = reviews.stars_review.iloc[orig_docs]
    print(np.bincount(orig_stars, minlength=5)[1:])
    all_stars.append(orig_stars)
#%%
show_cluster(np.argmin(np.abs(np.mean(all_stars, axis=1) - 3)))
#%%
closest = np.argmin(dists_to_centers, axis=1)
np.argsort(np.bincount(closest))
#%%
dist_to_closest_cluster = np.min(dists_to_centers, axis=1)
is_close = dist_to_closest_cluster < np.median(dist_to_closest_cluster)
#[sents_2[idx] for idx in np.argsort(dist_to_closest_cluster)[-50:]]
#%%
#%%
omit_clusters = []
# Train LMs on each cluster
from suggestion.util import dump_kenlm
for cluster_idx in range(n_clusters):
    sents_in_cluster = np.flatnonzero((closest == cluster_idx) & is_close)
    if len(sents_in_cluster) < 50:
        omit_clusters.append(cluster_idx)
    print(cluster_idx)
    dump_kenlm(f'yelp_bigclust_{cluster_idx}', [sents_2[idx] for idx in sents_in_cluster])
#%%
clusters_to_use = np.zeros(n_clusters, dtype=bool)
clusters_to_use.fill(True)
clusters_to_use[omit_clusters] = False
clusters_to_use = np.flatnonzero(clusters_to_use)
#%%
unique_starts = [x.split() for x in sorted({' '.join(sent.split()[:5]) for sent in sents_2})]
#%%
from suggestion import lang_model
from suggestion.paths import paths

scores_by_cluster = []
for cluster_idx in tqdm.tqdm(clusters_to_use):
    model = lang_model.Model.from_basename(paths.model_basename(f'yelp_bigclust_{cluster_idx}'))
    scores_by_cluster.append([model.score_seq(model.bos_state, k)[0] for k in unique_starts])
#%%
sbc = np.array(scores_by_cluster).T
#%%
from scipy.misc import logsumexp

likelihood_bias = logsumexp(sbc, axis=1, keepdims=True)
sbc2 = sbc - .85 * likelihood_bias
#%%
sbc_argsort = np.argsort(sbc2, axis=0)
#%%
import contextlib
with open('cluster_starts.txt', 'w') as f, contextlib.redirect_stdout(f):
    for cluster_idx in range(len(clusters_to_use)):
        print(clusters_to_use[cluster_idx])
        for idx in sbc_argsort[:,cluster_idx][-10:]:
            print(' '.join(unique_starts[idx]))
        print()
#%%
#Ypred = mbk_wordvecs.predict(
#    clustering.normalize_vecs(
#    vectorizer.transform(labeled_sents).dot(projection_mat)))

#%%
cluster_token = '<C{}>'.format
star_token = '<S{}>'.format
def sentidx_token(idx):
    return '<start>' if idx == 0 else '<mid>'
#%%
segments = []
for idx in tqdm.trange(len(reviews)):
    if reviews.is_train.iloc[idx]:
        segment = 0#'train'
    elif reviews.is_valid.iloc[idx]:
        segment = 1#'valid'
    else:
        segment = 2#'test'
    segments.append(segment)

#%%
sources = [open(f'src_{kind}_mini.txt', 'w') for kind in 'train valid test'.split()]
tgts = [open(f'tgt_{kind}_mini.txt', 'w') for kind in 'train valid test'.split()]

for idx in tqdm.trange(1000):#len(dense_vecs_2)):
    review_idx = doc_idx_0[orig_indices_2[idx]]
    segment = segments[review_idx]
    src_sent = f'{cluster_token(closest[idx])} {star_token(reviews.stars_review.iloc[review_idx])}\n'
    sources[segment].write(src_sent)
    tgts[segment].write(sents_2[idx] + '\n')
del sources
del tgts

#%%
# How long would it take to do all the ngrams ourselves?
import nltk
from collections import Counter
import tqdm
vocab = Counter(tok for toks in tqdm.tqdm(reviews.tokenized) for tok in toks.lower().split())
#%%
id2word = ['<unk>', '<s>', '</s>'] + [tok for tok, count in vocab.most_common()]
word2id = {tok: idx for idx, tok in enumerate(id2word)}
#%%
ctr = 0
tables = [{} for i in range(5)]
for toks in tqdm.tqdm(reviews.tokenized.iloc[:10], desc="Counting ngrams"):
    indices = [word2id[tok] for tok in toks.lower().split()]
    for ngram in nltk.ngrams(indices, 5, pad_left=True, pad_right=True, left_pad_symbol=1, right_pad_symbol=2):
        backoff = None
        for gram in range(4, -1, -1):
            print(ngram[gram:], ngram[gram])
            key = (backoff, ngram[gram])
            backoff = ctr
        assert False

#%%
all_indices = ([word2id[tok] for tok in toks.lower().split()] for toks in tqdm.tqdm(reviews.tokenized, desc="Counting ngrams"))
bigrams = Counter(tuple(bigram) for indices in all_indices for bigram in nltk.ngrams(indices, 2, pad_left=True, pad_right=True, left_pad_symbol=1, right_pad_symbol=2))
#%%
pruned_bigrams = [(bigram, count) for bigram, count in bigrams.items() if count >= 2]