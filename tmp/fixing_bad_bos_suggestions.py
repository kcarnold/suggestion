# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:36:56 2017

@author: kcarnold
"""

from suggestion import suggestion_generator, lang_model
from suggestion.paths import paths
import numpy as np
from scipy.misc import logsumexp

#%%
clizer = suggestion_generator.clizer
n_clusters = clizer.n_clusters
models = [lang_model.Model.from_basename(paths.model_basename('cluster_{}'.format(cluster_idx))) for cluster_idx in range(n_clusters)]
#%%
has_unks = np.array([[any(model.model.vocab_index(tok) == 0 for tok in toks) for model in models] for toks in clizer.unique_starts])
#%%
omit2 = np.flatnonzero(np.sum(has_unks, axis=1))
#%%
import re
has_review = np.array([bool(re.search(r'\breview(er|ed)?s?\b|\bstars?\b', ' '.join(toks))) for toks in clizer.unique_starts])
#%%
scores_by_cluster = clizer.scores_by_cluster.copy()
likelihood_bias = logsumexp(scores_by_cluster, axis=1, keepdims=True)
scores_by_cluster -= likelihood_bias
#scores_by_cluster[suggested_already] = -np.inf
scores_by_cluster[omit2] = -np.inf
#scores_by_cluster[has_review] = -np.inf
scores_by_cluster[clizer.omit] = -np.inf
most_distinctive = np.argmax(scores_by_cluster, axis=0)
#scores_by_cluster[most_distinctive] = -np.inf
#most_distinctive = np.argmax(scores_by_cluster, axis=0)


for i in range(n_clusters):
    print(i, ' '.join(clizer.unique_starts[most_distinctive[i]]))