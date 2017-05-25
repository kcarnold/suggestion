# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:13:54 2017

@author: kcarnold
"""

import numpy as np
from suggestion import clustering

#%%
cnnb = clustering.ConceptNetNumberBatch.load()
#%%
clustering.cnnb = cnnb
#%%
from suggestion.paths import paths
import pandas as pd
import os

data = pd.read_pickle(os.path.join(paths.parent, 'yelp_preproc/all_data.pkl'))
reviews = data['data'].reset_index(drop=True)
#%%
sentence_docs = []
sentences = []
for i, doc in enumerate(reviews.tokenized):
    for sent in doc.split('\n'):
        sentence_docs.append(i)
        sentences.append(sent)
#%%
sentence_lengths = np.array([len(sent.split()) for sent in sentences])
min_sent_len, max_sent_len = np.percentile(sentence_lengths, [25, 75])
is_reasonable_length = (min_sent_len <= sentence_lengths) & (sentence_lengths <= max_sent_len)
#%%
sentence_docs = [sentence_docs[idx] for idx in np.flatnonzero(is_reasonable_length)]
sentences = [sentences[idx] for idx in np.flatnonzero(is_reasonable_length)]

#%% Vectorize the sentences

vectorizer, raw_vecs = clustering.get_vectorizer(sentences)
projection_mat = clustering.get_projection_mat(vectorizer)
vecs = raw_vecs.dot(projection_mat)
#%%
min_norm = 0.5
norms = np.linalg.norm(vecs, axis=1)
large_enough = norms > min_norm
vecs = vecs[large_enough] / norms[large_enough][:,None]
sentences_filt = [sentences[i] for i in np.flatnonzero(large_enough)]
sentence_docs_filt = [sentence_docs[i] for i in np.flatnonzero(large_enough)]
#%%
import tqdm
last_doc = None
last_vec = None
pair_doc = []
vecs_a = []
vecs_b = []
for i, doc_idx in enumerate(tqdm.tqdm(sentence_docs_filt)):
    vec = vecs[i]
    if doc_idx == last_doc:
        pair_doc.append(i)
        vecs_a.append(last_vec)
        vecs_b.append(vec)
    last_doc = doc_idx
    last_vec = vec
#%%
vecs_a = np.array(vecs_a)
vecs_b = np.array(vecs_b)
#%%
raw_norms_vecsA = np.linalg.norm(vecs_a, axis=1)
raw_norms_vecsB = np.linalg.norm(vecs_b, axis=1)
#%% (a-b)**2 = a**2 + b**2 - 2*a*b. And both A and B have unit norm. So...
vec_pair_distances = np.sqrt(2-np.sum(vecs_a*vecs_b, axis=1))
close_threshold = np.median(vec_pair_distances)
#%%
close_indices = np.flatnonzero(vec_pair_distances < close_threshold)
far_indices = np.flatnonzero(vec_pair_distances >= close_threshold)
#%%
vec_pairs = np.concatenate((vecs_a, vecs_b), axis=1)
#%%
n_clusters = 10
random_state = 0
from sklearn.cluster import MiniBatchKMeans
mbk_close = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=random_state)
mbk_close.fit(vec_pairs[close_indices])
#%%
mbk_far = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=random_state)
mbk_far.fit(vec_pairs[far_indices])
#%%
cluster_dists_close = mbk_close.transform(vec_pairs)
cluster_dists_far = mbk_far.transform(vec_pairs)
#%%
cluster_dists = cluster_dists_close
for c in range(cluster_dists.shape[1]):
    print(c)
    for i in np.argsort(cluster_dists[:,c])[:10]:
        print(sentences_filt[pair_doc[i]-1].replace('\n', ' '), '|||', sentences_filt[pair_doc[i]].replace('\n', ' '))
    print()
#%%
from scipy.spatial.distance import cdist, euclidean
#%%
#%%
from matplotlib import pyplot as plt
import seaborn as sns
#%%
cluster_vecsA = mbk_far.cluster_centers_[:,:300]
cluster_vecsB = mbk_far.cluster_centers_[:,300:]
norms_vecsA = np.linalg.norm(cluster_vecsA, axis=1)
norms_vecsB = np.linalg.norm(cluster_vecsB, axis=1)
#sns.distplot(norms_)
with sns.axes_style('white'):
    sns.jointplot(x=norms_vecsA, y=norms_vecsB, color='k')
#%%
pair_distances_close = np.sqrt(2-np.sum(mbk_close.cluster_centers_[:,:300]*mbk_close.cluster_centers_[:,300:], axis=1))
pair_distances_far = np.sqrt(2-np.sum(mbk_far.cluster_centers_[:,:300]*mbk_far.cluster_centers_[:,300:], axis=1))
#%%
