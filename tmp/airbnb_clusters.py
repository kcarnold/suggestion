# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:20:59 2017

@author: kcarnold
"""

import glob
import pandas as pd
import numpy as np
#%%
raw_listings = pd.concat([pd.read_csv(x) for x in glob.glob('/Data/Airbnb/*/listings*')], axis=0)
#%%
listings = raw_listings[raw_listings.space.str.len() > 10]
#%%
import nltk
import tqdm
sents = [nltk.sent_tokenize(txt) for txt in tqdm.tqdm(listings.space)]
#%%
sents_0 = [sent for doc in sents for sent in doc]
sent_lens = np.array([len(sent) for sent in sents_0])
#%%
sents_1 = sorted(set(sents_0))
#%%
from collections import defaultdict
def compute_sent_indices(sents_by_doc):
    sent_indices = defaultdict(list)
    for doc_idx, doc in enumerate(sents_by_doc):
        for sent_idx, sent in enumerate(doc):
            sent_indices[sent].append(sent_idx)
    return dict(sent_indices)
sent_indices = compute_sent_indices(sents)

#%%# Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5, max_df=.75, stop_words='english')
vecs_1 = vectorizer.fit_transform(sents_1)
# Note: The vectorizer normalized the vectors.

#%%
n_clusters = 128
random_state = 0
from sklearn.cluster import MiniBatchKMeans
from suggestion import clustering
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
#projection_mat = -np.log(wordfreqs_for_sklearn_vocab)[:,None] * cnnb_vecs_for_sklearn_vocab
projection_mat = cnnb_vecs_for_sklearn_vocab
#%%
dense_vecs_1 = vecs_1.dot(projection_mat)

#%% Normalize, and filter by norm.
min_norm = .5
norms_1 = np.linalg.norm(dense_vecs_1, axis=1)
indices_2 = np.flatnonzero(norms_1 > min_norm)
dense_vecs_2 = dense_vecs_1[indices_2] / norms_1[indices_2][:,None]
sents_2 = [sents_1[i] for i in indices_2]
#%%
n_clusters = 128
mbk_wordvecs = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=random_state)
mbk_wordvecs.fit(dense_vecs_2)
#%%
dists_to_centers = mbk_wordvecs.transform(dense_vecs_2)
#%%
def show_cluster(idx):
    close_indices = np.argsort(dists_to_centers[:,idx])[:10]
    for idx in close_indices:
        print(sents_2[idx])
#%%
for i in range(dists_to_centers.shape[1]):
    print()
    print(i)
    close_indices = np.argsort(dists_to_centers[:,i])[:10]
    for idx in close_indices:
        print(sents_2[idx])
#    orig_docs = doc_idx_0[orig_indices_2[close_indices]]
#    orig_stars = reviews.stars_review.iloc[orig_docs]
#    print(np.bincount(orig_stars, minlength=5)[1:])
#    all_stars.append(orig_stars)
#%%
closest = np.argmin(dists_to_centers, axis=1)
np.argsort(np.bincount(closest))
#%%
dist_to_closest_cluster = np.min(dists_to_centers, axis=1)
is_close = dist_to_closest_cluster < np.median(dist_to_closest_cluster)
#%%
from sklearn.naive_bayes import BernoulliNB
#%%
vecs_2 = vecs_1[indices_2]
#%%
indices_3 = np.flatnonzero(is_close)
sents_3 = [sents_2[idx] for idx in indices_3]
#%%
vectorizer2 = TfidfVectorizer(min_df=5, max_df=.75, ngram_range=(1,2), stop_words=None)
vecs_2 = vectorizer2.fit_transform([' '.join(sent.split()[:5]) for sent in sents_3])
#%%
#X = vecs_2[is_close]
X = vecs_2
y = closest[is_close]
#%%
clf = BernoulliNB().fit(X, y)
#%%
probs = clf.predict_proba(X)
#%%
show_cluster(1)
#%%
import contextlib
with open('airbnb_clusters.txt', 'w') as f, contextlib.redirect_stdout(f):
    for class_idx, cluster_idx in enumerate(clf.classes_):
        print(cluster_idx)
#        show_cluster(cluster_idx)
        for i in np.argsort(probs[:,class_idx])[-10:]:
            print(' '.join(sents_3[i].split()[:10]))
        print()
        print()
#%%
sentnum_y = np.array([sent_indices[sent][0] for sent in sents_3])
sentnum_y_enc = np.where(sentnum_y > 2, 2, sentnum_y)
#%%
sentnum_clf = BernoulliNB().fit(X, sentnum_y_enc)
#%%
sentnum_probs = sentnum_clf.predict_proba(X)
#%%
for cls_idx, cls in enumerate(sentnum_clf.classes_):
    for i in np.argsort(sentnum_probs[:,cls_idx])[-10:]:
        print(' '.join(sents_3[i].split()[:10]))
    print()
#%%
cluster_data = []
import contextlib
with open('airbnb_starts.txt', 'w') as f, contextlib.redirect_stdout(f):
    for class_idx, cluster_idx in enumerate(clf.classes_):
        indices = np.argsort(probs[:,class_idx])[-10:]
        cluster_sentnum_probs = np.mean(sentnum_probs[indices], axis=0)
        s1, s2, sN = cluster_sentnum_probs
        print(f'Cluster {cluster_idx} p(sent_num) = [{s1:.2f} {s2:.2f} {sN:.2f}]')
#        show_cluster(cluster_idx)
        datum = []
        for i in indices:
            print(f'{np.round(sentnum_probs[i],2)}', ' '.join(sents_3[i].split()[:10]))
            datum.append((sentnum_probs[i].tolist(), ' '.join(sents_3[i].split()[:10])))
        print()
        print()
        cluster_data.append(datum)
#%%
import json
json.dump(cluster_data, open('airbnb_cluster_data.json','w'))
#%%
airbnb_examples = np.random.RandomState(0).choice(listings.space, 10, replace=False).tolist()
json.dump(airbnb_examples, open('airbnb_examples.json', 'w'))