# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:46:57 2017

@author: kcarnold
"""

from megacomplete import data
import numpy as np
import scipy.sparse
#%%
sents = data.yelp_sents()
#%%
sent_lens = np.array([len(sent) for doc in sents for sent in doc])
min_sent_len, max_sent_len = np.percentile(sent_lens, [25, 75])
#%%
rs = np.random.RandomState(0)
reasonable_length_sents = [[sent for sent in doc if min_sent_len <= len(sent) <= max_sent_len] for doc in sents]
orig_sents_flat = [rs.choice(doc_sents) for doc_sents in reasonable_length_sents if doc_sents]
print('\n'.join(np.random.choice(orig_sents_flat, 10, replace=False)))
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5, max_df=.5, stop_words='english')
orig_vecs = vectorizer.fit_transform(orig_sents_flat)
#%%
vec_norms = scipy.sparse.linalg.norm(orig_vecs, axis=1)
indices_to_keep = np.flatnonzero(vec_norms)
vecs = orig_vecs[indices_to_keep]
sents_flat = [orig_sents_flat[i] for i in indices_to_keep]
#%%
print('\n'.join(np.random.choice(sents_flat, 10, replace=False)))
#%%
# Similarity
#import numpy as np
#
#sims = vecs * vectorizer.transform(['the service was great']).T
#sims_A = sims.A.ravel().copy()
#sims_A[sims_A > .999] = 0
#sims_argsort = np.argsort(sims_A)
#[sents_flat[i] for i in sims_argsort[-50:]]

#%%
from sklearn.cluster import MiniBatchKMeans
mbk = MiniBatchKMeans(init='k-means++', n_clusters=10, n_init=10)
clusters = mbk.fit_predict(vecs)
#%%
import numpy as np
np.bincount(clusters)
#%%
for c in range(np.max(clusters)+1):
    ss = np.flatnonzero(clusters == c)
    np.random.shuffle(ss)
    for i in ss[:10]:
        print(sents_flat[i])
    print()
#%%
cluster_dists = mbk.transform(vecs)
for c in range(cluster_dists.shape[1]):
    print(c)
    for i in np.argsort(cluster_dists[:,c])[:10]:
        print(i, sents_flat[i].replace('\n', ' '))
    print()

#%%
import subprocess
def dump_kenlm(model_name, tokenized_sentences):
    # Dump '\n'.join(' '.join-formatted tokenized reviews, without special markers,
    # to a file that KenLM can read, and build a model with it.
    with open('models/{}.txt'.format(model_name), 'w') as f:
        for toks in tokenized_sentences:
            print(toks.lower(), file=f)
    subprocess.run(['./scripts/make_model.sh', model_name])
#%%
# We used a subsample of sentences for making the clustering. Train the LMs on the full set, though.

# or not.
sentences_in_cluster = [[] for i in range(mbk.n_clusters)]
for i, c in enumerate(clusters):
    sentences_in_cluster[c].append(orig_sents_flat[i])
#%%
[len(c) for c in sentences_in_cluster]
#%%
for cluster_idx, cluster in enumerate(sentences_in_cluster):
    print(cluster_idx)
    dump_kenlm('cluster_{}'.format(cluster_idx), [s.lower() for s in cluster])
#%%
from suggestion import suggestion_generator, paths
models = [suggestion_generator.Model.from_basename(paths.paths.model_basename('cluster_{}'.format(cluster_idx))) for cluster_idx in range(mbk.n_clusters)]
#%%
