# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:05:43 2017

@author: kcarnold
"""

import numpy as np
import pandas as pd
import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import contextlib

#%%
from suggestion.analyzers import load_reviews
reviews = load_reviews()
#%%
sents_0 = [sent for doc in reviews.tokenized for sent in doc.lower().split('\n')]
doc_idx_0 = np.array([doc_idx for doc_idx, doc in enumerate(reviews.tokenized) for sent in doc.lower().split('\n')])
#%%
sent_idx_0 = np.array([sent_idx for doc in reviews.tokenized for sent_idx, sent in enumerate(doc.lower().split('\n'))])
#%%
sent_lens = np.array([len(sent.split()) for sent in sents_0])
min_sent_len, max_sent_len = np.percentile(sent_lens, [25, 75])
indices_1 = np.flatnonzero((min_sent_len <= sent_lens) & (sent_lens <= max_sent_len))
sents_1 = [sents_0[i] for i in indices_1]
#%%# Vectorize
vectorizer = TfidfVectorizer(min_df=5, max_df=.75, stop_words='english', ngram_range=(1,2))
vecs_1 = vectorizer.fit_transform(sents_1)
# Note: The vectorizer normalized the vectors.

#%%
random_state = 0
from suggestion import clustering
cnnb = clustering.ConceptNetNumberBatch.load()

#%%
sklearn_vocab = vectorizer.get_feature_names()
def get_or_zero(cnnb, item):
    try:
        return cnnb[item.replace(' ', '_')]
    except KeyError:
        return np.zeros(cnnb.ndim)
cnnb_vecs_for_sklearn_vocab = np.array([get_or_zero(cnnb, word) for word in sklearn_vocab])
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
dists_to_centers_2 = mbk_wordvecs.transform(dense_vecs_2)
#%%
dists_to_centers_argsort_2 = np.argsort(dists_to_centers_2, axis=0)
#%%
def show_cluster(idx):
    close_indices = np.argsort(dists_to_centers_2[:,idx])[:10]
    for idx in close_indices:
        print(sents_2[idx])
#%%
for i in range(dists_to_centers_2.shape[1]):
    print()
    print(i)
    close_indices = np.argsort(dists_to_centers_2[:,i])[:10]
    for idx in close_indices:
        print(sents_2[idx])
#    orig_docs = doc_idx_0[orig_indices_2[close_indices]]
#    orig_stars = reviews.stars_review.iloc[orig_docs]
#    print(np.bincount(orig_stars, minlength=5)[1:])
#    all_stars.append(orig_stars)
#%%
closest_cluster_2 = np.argmin(dists_to_centers_2, axis=1)
np.argsort(np.bincount(closest_cluster_2))
#%%
dist_to_closest_cluster = np.min(dists_to_centers_2, axis=1)
is_close = dist_to_closest_cluster < np.median(dist_to_closest_cluster)

#%%
#
# Find cluster-distinguishing sentence openers.
#
#%% Consider only sentences that are near their corresponding clusters (more clusters could reduce these)
indices_3 = np.flatnonzero(is_close)
sents_3 = [sents_2[idx] for idx in indices_3]
#%% Re-vectorize just for sentence openers, without stopwords, but tighter cutoffs.
opener_vectorizer = TfidfVectorizer(min_df=5, max_df=.75, ngram_range=(1,2), stop_words=None)
opener_vecs_3 = opener_vectorizer.fit_transform([' '.join(sent.split()[:5]) for sent in sents_3])
#%%
closest_cluster_3 = closest_cluster_2[indices_3]
#%% Train a classifier to predict cluster from opener-vec
opener_clf = BernoulliNB().fit(opener_vecs_3, closest_cluster_3)
#%%
opener_probs = opener_clf.predict_proba(opener_vecs_3)
#%%
show_cluster(1)
#%%
with open('yelp_newclusters.txt', 'w') as f, contextlib.redirect_stdout(f):
    for class_idx, cluster_idx in enumerate(opener_clf.classes_):
        print(cluster_idx)
#        show_cluster(cluster_idx)
        for i in np.argsort(opener_probs[:,class_idx])[-10:]:
            print(' '.join(sents_3[i].split()[:10]))
        print()
        print()
#%% Now try a couple ways of predicting sentiment. First, are some clusters just positive or negative?
doc_idx_3 = doc_idx_0[indices_1[indices_2[indices_3]]]
stars_3 = np.array(reviews.stars_review.iloc[doc_idx_3])
#%%
stars_per_cluster = np.array([np.bincount(stars_3[np.flatnonzero(closest_cluster_3 == cluster_idx)] - 1, minlength=5) for cluster_idx in range(n_clusters)])+100
star_dist = stars_per_cluster / np.sum(stars_per_cluster, axis=1, keepdims=True)
star_dist_argsort = np.argsort(star_dist, axis=0)
#%%
#np.argsort(star_dist[:,0])
#star_dist_argsort[:, 0]
#%%
for star in range(5):
    print(f'{star+1}-star')
    for cluster_idx in star_dist_argsort[:,star][-3:]:
        print(f'- cluster {cluster_idx}')
        for idx in dists_to_centers_argsort_2[:,cluster_idx][:3]:
            print(sents_2[idx])
        print()
    print()

#%% Opinionated vs unopinionated clusters. Find the least opinionated.
#for cluster_idx in np.argsort(np.min(star_dist, axis=1))[:3]:
from scipy.special import entr
for cluster_idx in np.argsort(entr(star_dist).sum(axis=1))[-3:]:
    print(cluster_idx)
    show_cluster(cluster_idx)
    print()
    print()


#[sents_3[i] for i in np.argsort(sentiment_probs_3[:,4])[-10:]]
#%%
#%%
# Grr, some of those openings indicate places. Fix that.
place_labels = reviews.city.str.cat(reviews.state, sep=', ')
place_indices = {place: idx for idx, place in enumerate(sorted(place_labels.unique()))}
place_label_as_idx = np.array([place_indices[lbl] for lbl in place_labels])
#%%
geog_clf = BernoulliNB().fit(opener_vecs_3, place_label_as_idx[doc_idx_3])
#%%
geog_probs = geog_clf.predict_proba(opener_vecs_3)
#%%
geog_distinctiveness = np.max(geog_probs, axis=1)
#%%
def summarize_argsort(labels, scores, n=10, n_mid=10, show=lambda label, score: print(f'{score:.2f} {label}')):
    argsort = np.argsort(scores)
    print('smallest')
    for i in argsort[:n]:
        show(labels[i], scores[i])

    if n_mid:
        mid_idx = len(argsort) // 2
        low_mid_idx = mid_idx - n // 2
        high_mid_idx = low_mid_idx + n
        print('\nmiddle')
        for i in argsort[low_mid_idx:high_mid_idx]:
            show(labels[i], scores[i])

    print('\nlargest')
    for i in argsort[-n:]:
        show(labels[i], scores[i])

summarize_argsort(sents_3, geog_distinctiveness)
#%% Ok, any phrase with geographical distinctiveness > .5 is taboo.
#%% Try to predict sentence number
sent_idx_3 = sent_idx_0[indices_1[indices_2[indices_3]]]
#%%
sent_idx_clf = BernoulliNB().fit(opener_vecs_3, np.minimum(sent_idx_3, 2))
sent_idx_probs = sent_idx_clf.predict_proba(opener_vecs_3)
#%%
summarize_argsort(sents_3, sent_idx_probs[:,1])
np.sum(sent_idx_probs > .5, axis=0)

#%%
#%% Goal: offer suggestions that are (1) positive and (2) about a variety of topics.
# First, what are the most positive sentence-starters?
sentiment_clf = BernoulliNB(alpha=5.).fit(opener_vecs_3, stars_3)
sentiment_probs_3 = sentiment_clf.predict_proba(opener_vecs_3) # careful not to overfit!
#%%
vecs_for_sim_3 = dense_vecs_2[indices_3]
#%%
# Pick some diverse openers.
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
vecs_for_sim = opener_vecs_3

import re
cant_type = re.compile(r'[^\-a-z., !\']')
#%%
by_stars = []
for stars_tgt in tqdm.trange(5):
    by_sent = []
    for sent_num in range(3):
        opener_indices = []

        taboo = set()
        for idx in np.argsort(sentiment_probs_3[:,stars_tgt])[::-1]:
            sent = sents_3[idx]
            if cant_type.search(sent) is not None:
                continue
            toks = sent.split()
            orig_star = stars_3[idx] - 1
            if orig_star != stars_tgt:
                continue
            if geog_distinctiveness[idx] > .5:
                continue
            if min(2, sent_idx_3[idx]) != sent_num:
                continue
            if sent_idx_probs[idx, sent_num] < .75:
                continue
            starter = ' '.join(toks[:3])
            if starter in taboo:
                continue
            if len(opener_indices) > 0:
                closest_existing_pt, dist_to_closest = pairwise_distances_argmin_min(vecs_for_sim[idx:idx+1], vecs_for_sim[opener_indices])
                # They're arrays because Y is an array; get the scalars.
                closest_existing_pt = opener_indices[closest_existing_pt.item()]
                dist_to_closest = dist_to_closest.item()
                # Make sure this new one isn't too close
                if dist_to_closest < 1.2:
                    continue
            else:
                dist_to_closest = 0
        #        print(f'closest: {closest_existing_pt} {sents_3[closest_existing_pt]} {dist_to_closest:.2f}')
            opener_indices.append(idx)
            taboo.add(starter)
#            print(f'{idx:6d} {dist_to_closest:.2f} {stars_3[idx]} {sent_idx_3[idx]} {sentiment_probs_3[idx,stars_tgt]:.2f}', ' '.join(sents_3[idx].split()[:5]))
            if len(opener_indices) == 100:
                break
        by_sent.append(opener_indices)
    by_stars.append(by_sent)
#%%
by_stars_sents = [
    [
     [sents_3[idx] for idx in indices]
     for indices in by_sent]
    for by_sent in by_stars]
#%%
import json
json.dump(by_stars_sents, open('yelp_sentiment_starters.json','w'), indent=2)
