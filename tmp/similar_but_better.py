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
from scipy.spatial.distance import pdist
import tqdm
import pandas as pd

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
#%%
j = 0
for i, review in enumerate(tqdm.tqdm(reviews.tokenized)):
    meta = {k: reviews[k].iloc[i] for k in 'stars_review stars_biz age_months is_best'.split()}
    for sent in review.split('\n'):
        res[j]['sent'] = sent
        res[j].update(meta)
        j += 1
#%%
quals = pd.DataFrame(res)
#quals.to_csv(str(paths.parent / 'data' / 'sentence_quality.csv'))
#%%
with open(paths.parent / 'data' / 'sentence_quality.pkl', 'wb') as f:
    pickle.dump(res, f, -1)
#%%
with open(paths.parent / 'data' / 'sentence_quality.pkl', 'rb') as f:
    res = pickle.load(f)

#%%
import statsmodels.api as sm
import statsmodels.formula.api as smf
results = smf.ols('np.log1p(votes_useful) ~ len_words + unifreq_mean + unifreq_std + wordpair_mean + wordpair_std + age_months', data=quals).fit()
#%%
print(results.summary())

#%%
quals_pred = pd.read_csv(str(paths.parent / 'data' / 'sentence_quality_with_predictions.dat'))
#%%
quals_pred.predicted_votes_useful
#%%
quals['pred'] = quals_pred.predicted_votes_useful
#%%
best_sents = quals[
        #(quals.pred > np.nanpercentile(quals.pred, 90)) &
        (quals.is_best)].sent.tolist()
#%%
sent_vecs = suggestion_generator.clizer.vectorize_sents(best_sents)
#%%
normed = suggestion_generator.clustering.normalize_vecs(sent_vecs)
orig_mags = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
#%% ok try it out.
query = 'wide selection'
vec = suggestion_generator.clustering.normalize_vecs(suggestion_generator.clizer.vectorize_sents([query]))[0]
dotp = normed @ vec
#dotp[dotp>.75] = 0
[best_sents[i] for i in np.argsort(dotp)[-10:][::-1]]
