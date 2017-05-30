# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:36:36 2017

@author: kcarnold
"""

import pandas as pd
import pickle
import numpy as np
#%% Load all the reviews.
data = pickle.load(open('yelp_preproc/all_data.pkl','rb'))
vocab, counts = data['vocab']
reviews = data['data'].reset_index(drop=True)
del data
#%%
sentences = [sent for doc in reviews.tokenized for sent in doc.split('\n')]
#%%
from collections import namedtuple
YelpSent = namedtuple('YelpSent', 'words tags')
tagged_sentences = [YelpSent(sent.lower().split(), [i]) for i, sent in enumerate(sentences)]
#%%

from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

model = Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores)
#%%
model.build_vocab(tagged_sentences)
#%%
import random
random.shuffle(tagged_sentences)
#%%
model.min_alpha = model.alpha = 0.025
model.train(tagged_sentences)
#%%
doc_id = np.random.randint(model.docvecs.count)
sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)  # get *all* similar documents
print(u'TARGET (%d): «%s»\n' % (doc_id, ' '.join(tagged_sentences[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(tagged_sentences[sims[index][0]].words)))
#%%
while True:
    word = random.choice(model.wv.index2word)
    if model.wv.vocab[word].count > 100:
        break
print(word)
model.most_similar(word, topn=20)

#%%
vecs = model.docvecs.doctag_syn0
#%%
sent_lens_chr = np.array([len(sent) for sent in sentences])
#%%
min_sent_len, max_sent_len = np.percentile(sent_lens_chr, [25, 75])
is_reasonable_length = (min_sent_len <= sent_lens_chr) & (sent_lens_chr <= max_sent_len)
rl_indices = np.flatnonzero(is_reasonable_length)
vecs = model.docvecs.doctag_syn0[is_reasonable_length]
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
#%%
from sklearn.cluster import MiniBatchKMeans
mbk = MiniBatchKMeans(init='k-means++', n_clusters=10, n_init=10)
clusters = mbk.fit_predict(vecs)
#%%
cluster_dists = mbk.transform(vecs)
for c in range(cluster_dists.shape[1]):
    print(c)
    for i in np.argsort(cluster_dists[:,c])[:10]:
        print(i, sentences[rl_indices[i]].replace('\n', ' '))
    print()
