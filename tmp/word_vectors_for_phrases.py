# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:44:23 2017

@author: kcarnold
"""

import numpy as np
from suggestion import clustering
from suggestion import suggestion_generator

#%%
cnnb = clustering.ConceptNetNumberBatch.load()
#%%
vec = cnnb['wait'] + cnnb['server']
sims = cnnb.vecs @ vec
#%%
[cnnb.id2term[i] for i in np.argsort(sims)[::-1][:50]]
#%%

import pickle

gs_w2v = pickle.load(open('/Users/kcarnold/first_yelp_w2v.pkl', 'rb'))

#%%
(cnnb['empty']) @ (cnnb['full'])