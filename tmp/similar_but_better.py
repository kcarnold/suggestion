# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:32:22 2017

@author: kcarnold
"""

from suggestion import analyzers
from suggestion import suggestion_generator
from suggestion.paths import paths
import pickle
import numpy as np
#%%
reviews = analyzers.load_reviews()
#%%
best_sents = [sent for sents in reviews[reviews.is_best].tokenized for sent in sents.split('\n')]
#%%
sent_vecs = suggestion_generator.clizer.vectorize_sents(best_sents)
#%%
normed = suggestion_generator.clustering.normalize_vecs(sent_vecs)
orig_mags = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
#%% ok try it out.
query = 'food was good'
vec = suggestion_generator.clustering.normalize_vecs(suggestion_generator.clizer.vectorize_sents([query]))[0]
dotp = normed @ vec
dotp[dotp>.75] = 0
best_sents[np.argmax(dotp)]

#%% Nope, can make poor-quality suggestions. Lot's look at distribution of:
# sent length
# unigram word freq mean and std
# word-pair dist mean and std
wordfreq_analyzer = analyzers.WordFreqAnalyzer.build()
#%%
wordpair_analyzer = pickle.load(open(paths.models / 'wordpair_analyzer.pkl', 'rb'))
#%%
from scipy.spatial.distance import pdist

import tqdm
res = []
for i, review in enumerate(tqdm.tqdm(reviews.tokenized)):
    datum_base = {k: reviews[k].iloc[i] for k in ['votes_funny', 'votes_useful', 'votes_cool']}
    for sent in review.split('\n'):
        datum = datum_base.copy()
        toks = sent.split()

        # Length
        datum['len_words'] = len(toks)
        datum['len_chars'] = len(sent)

        # Frequency
        indices = wordfreq_analyzer.lookup_indices(toks)
        if len(indices):
            unifreqs = wordfreq_analyzer.log_freqs[indices]
            datum['unifreq_mean'] = np.mean(unifreqs)
            datum['unifreq_std'] = np.std(unifreqs)

        # Pairs
        word_vecs = wordpair_analyzer.vectorizer.transform([sent])
        if len(word_vecs.indices) >= 2:
            #word_vecs.data[:,None] *
            projected = wordpair_analyzer.projection_mat[word_vecs.indices]
            # These are already normed.
            dists = pdist(projected)
            datum['wordpair_mean'] = np.mean(dists)
            datum['wordpair_std'] = np.std(dists)
        res.append(datum)