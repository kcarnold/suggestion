# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:53:40 2017

@author: kcarnold
"""

import numpy as np
import pandas as pd
from suggestion import clustering
from suggestion.util import dump_kenlm
#%%
sents = clustering.filter_reasonable_length_sents(clustering.get_all_sents())
#%%
LM_seeds = 'food service location ambiance value'.split()
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5, max_df=.5, stop_words='english')
all_vecs = vectorizer.fit_transform(sents)
#%%
vocab_indices = [vectorizer.vocabulary_[w] for w in LM_seeds]
#%%
sents_by_cluster = [all_vecs[:,idx].nonzero()[0] for idx in vocab_indices]
#%%
for word, sent_indices in zip(LM_seeds, sents_by_cluster):
    print(word)
    dump_kenlm(f'tmp_{word}_0', [sents[idx] for idx in sent_indices])
#%%
from suggestion import lang_model
from suggestion.paths import paths
#%%
models = [lang_model.Model.from_basename(paths.model_basename(f'tmp_{word}_0')) for word in LM_seeds]
#%%
import tqdm
scores_by_cluster = np.array([[model.score_seq(model.bos_state, k)[0] for model in models] for k in tqdm.tqdm(sents, desc="Score sents")])
#%%
sbc_lmnorm = scores_by_cluster - np.mean(scores_by_cluster, axis=0)
#%%

from scipy.misc import logsumexp
sbc_lse = logsumexp(scores_by_cluster, axis=1, keepdims=True)
#%%
sbc = scores_by_cluster - 1 * sbc_lse

for cluster_idx, word in enumerate(LM_seeds):
    N = 0
    print(word)
    for idx in np.argsort(sbc[:,cluster_idx])[::-1]:
        if all_vecs[idx, vocab_indices[cluster_idx]] == 0 and cluster_idx == np.argmax(sbc[idx]):
            print(idx, sents[idx])
            N += 1
            if N == 10:
                break

    print()
