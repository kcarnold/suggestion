# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:31:07 2017

@author: kcarnold
"""

import numpy as np
from scipy.special import expit
from suggestion.paths import paths
from suggestion import lang_model
Model = lang_model.Model
#%%
PRELOAD_MODELS = '''
yelp_train
yelp_train-1star
yelp_train-2star
yelp_train-3star
yelp_train-4star
yelp_train-5star'''.split()
models = {name: Model.from_basename(paths.model_basename(name)) for name in PRELOAD_MODELS}
#%%

class LMClassifier:
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(weights, dtype=float)

    def get_state(self, toks, bos=False):
        models = self.models
        return [model.get_state(toks, bos=bos)[0] for model in models], np.zeros(len(models))

    def advance_state(self, state, tok):
        lm_states, scores = state
        new_lm_states = []
        score_deltas = np.empty(len(lm_states))
        for i, (lm_state, model) in enumerate(zip(lm_states, self.models)):
            new_lm_state, score_delta = model.advance_state(lm_state, tok)
            new_lm_states.append(new_lm_state)
            score_deltas[i] = score_delta
        new_state = new_lm_states, scores + score_deltas
        return new_state#, score_deltas

    def eval_posterior(self, state):
        lm_states, scores = state
        return expit(self.weights @ scores)

lmc = LMClassifier([models['yelp_train-1star'], models['yelp_train-5star']], [1., -1.])
for seq in ['this place was terrible', 'this place was amazing', 'this place was reasonably', 'my favorite', 'the only redeeming', 'i wanted to', 'we came here', 'service was slow', 'the service was very friendly']:
    print(seq)
    state = lmc.get_state(['<D>'], bos=True)
    hist = []
    for tok in seq.split():
        state = lmc.advance_state(state, tok)
        hist.append(lmc.eval_posterior(state))
    print(np.round(hist, 2))
#%%