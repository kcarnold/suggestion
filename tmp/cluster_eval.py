# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:04:31 2017

@author: kcarnold
"""

import numpy as np
import pandas as pd
from suggestion import clustering
#%%
sents = clustering.filter_reasonable_length_sents(clustering.get_all_sents())
#%%
rs = np.random.RandomState(0)
N = 500
indices = np.random.choice(len(sents), N, replace=False)
picked_sents = [sents[i] for i in indices]
#%%
pd.DataFrame({'idx': indices, 'sent': picked_sents}).to_excel('sentences_to_label.xlsx')
#%%
labeled = pd.read_excel('sentences_to_label.xlsx')
#%%
from collections import Counter
labels = [(row.sent, row.tags.split(',')) for row in labeled.itertuples() if isinstance(row.tags, str)]
all_tags = Counter(tag for sent, tags in labels for tag in tags)
valid_tags = [tag for tag, count in all_tags.most_common() if count > 1]
def filter_labels(labels, valid_tags):
    ret = []
    for sent, tags in labels:
        tags = [tag for tag in tags if tag in valid_tags]
        if len(tags) == 0:
            continue
        ret.append((sent, tags))
    return ret
labels = filter_labels(labels, valid_tags)
labeled_sents = [sent for sent, tags in labels]
#%%
def sample_labeling(rs):
    ret = []
    for sent, tags in labels:
        if len(tags) == 1:
            tag = tags[0]
        else:
            tag = tags[rs.choice(len(tags))]
        ret.append(valid_tags.index(tag))
    return ret

#%%
from sklearn import metrics
def score_clustering(Ypred):
    rand_scores = [metrics.adjusted_rand_score(
        np.array(sample_labeling(np.random.RandomState(i))), Ypred)
        for i in range(50)]
    return np.mean(rand_scores)

#%%
from suggestion import suggestion_generator
clizer = suggestion_generator.clizer
#%%
Ypred = clizer.clusterer.predict(
    clustering.normalize_vecs(
    clizer.vectorize_sents(labeled_sents)))
score_clustering(Ypred)
# -> 0.201 with the clizer we've been using.
#%%# Try with raw counts.
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5, max_df=.5, stop_words='english')
all_vecs = vectorizer.fit_transform(sents)
#%%
# The vectorizer normalized the vectors.
#%%
n_clusters = 10
random_state = 0
from sklearn.cluster import MiniBatchKMeans
mbk_raw = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=random_state)
mbk_raw.fit(all_vecs)
#%%
score_clustering(mbk_raw.predict(vectorizer.transform(labeled_sents)))
# -> 0.042. So the word vectors help a lot.
# -> 0.046 with no stopwords.
#%% # Reproduce the precomputed clizer result.
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
vecs = all_vecs.dot(projection_mat)
filtered_sents, projected_vecs = clustering.filter_by_norm(vecs, sents)
#%%
for n_clusters in [8, 9, 10, 11, 12]:
    mbk_wordvecs = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=random_state)
    mbk_wordvecs.fit(projected_vecs)
    Ypred = mbk_wordvecs.predict(
        clustering.normalize_vecs(
        vectorizer.transform(labeled_sents).dot(projection_mat)))
    print(n_clusters, score_clustering(Ypred))
#
# 5 0.193401513075
#7 0.190302764781
#8 0.181517530063
#9 0.205560507949
#10 0.228904621645
#11 0.187512586272
#12 0.167083634465
#13 0.199519145892
#15 0.17882461184
#20 0.126238412363
#25 0.140304316025
# -> best is 10, at 0.228, even better...


#%% Try without double-tf
projection_mat = cnnb_vecs_for_sklearn_vocab
vecs = all_vecs.dot(projection_mat)
filtered_sents, projected_vecs = clustering.filter_by_norm(vecs, sents)
#%%
for n_clusters in [8, 9, 10, 11, 12, 13]:
    mbk_wordvecs = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=random_state)
    mbk_wordvecs.fit(projected_vecs)
    Ypred = mbk_wordvecs.predict(
        clustering.normalize_vecs(
        vectorizer.transform(labeled_sents).dot(projection_mat)))
    print(n_clusters, score_clustering(Ypred))
#
#8 0.178914066284
#9 0.164575480198
#10 0.25810831653
#11 0.281801053129
#12 0.243877345134
# -> best is no-double-tf and 11 clusters.

# Wait, the above was with using tf-weighted vectors for the test sentences. If we use the same projection matrix, we get:
#8 0.123766052167
#9 0.130450138844
#10 0.199040954368
#11 0.229248081513
#12 0.173814626244
#13 0.135027740726

#%%
from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
dbscan.fit(projected_vecs)

#%% What explains the errors?
n_clusters = 10
mbk_wordvecs = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=random_state)
mbk_wordvecs.fit(projected_vecs)
Ypred = mbk_wordvecs.predict(
    clustering.normalize_vecs(
    vectorizer.transform(labeled_sents).dot(projection_mat)))
#%%
np.flatnonzero((Ypred == 0) & (np.array(sample_labeling(np.random.RandomState(0))) != 0))
#%%
#from sklearn.transform
#%% Could we just use a classifier?
def vectorize_labels(labels, all_tags):
    ret = np.zeros((len(labels), len(all_tags)))
    for i, (sent, tags) in enumerate(labels):
        for tag in tags:
            ret[i, all_tags.index(tag)] += 1
    return ret
sent_labels = vectorize_labels(labels, valid_tags)
#%%
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier().fit(vectorizer.transform(labeled_sents).dot(projection_mat), sent_labels)
#%%
test_sents = picked_sents[400:500]
test_vecs = vectorizer.transform(test_sents).dot(projection_mat)
preds = np.array(clf.predict_log_proba(test_vecs))[:,:,1].T
for i, sent in enumerate(test_sents):
    print(sent)
    print(','.join([valid_tags[t] for t in [np.argmax(preds[i])]]))
    print()

#%% OkNot terrible but not very good. Let's try that random-trees embedding.
from sklearn.ensemble import RandomTreesEmbedding
random_trees_xformer = RandomTreesEmbedding()
rt_embedding = random_trees_xformer.fit_transform(projected_vecs)
#%%
[filtered_sents[i] for i in np.argsort(rt_embedding.dot(random_trees_xformer.transform(vectorizer.transform([labeled_sents[11]]).dot(projection_mat)).T).A[:,0])[-10:]]
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
clf = MultiOutputClassifier(LogisticRegression()).fit(random_trees_xformer.transform(vectorizer.transform(labeled_sents).dot(projection_mat)), sent_labels)
test_sents = picked_sents[400:500]
test_vecs = random_trees_xformer.transform(vectorizer.transform(test_sents).dot(projection_mat))
preds = np.array(clf.predict_proba(test_vecs))[:,1,:]#.T
for i, sent in enumerate(test_sents):
    print(sent)
    print(','.join([valid_tags[t] for t in [np.argmax(preds[i])]]))
    print()
