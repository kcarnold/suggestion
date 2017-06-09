# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:31:07 2017

@author: kcarnold
"""

import numpy as np
from scipy.special import expit
from scipy.misc import logsumexp
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
models = {name: Model.from_basename(name, paths.model_basename(name)) for name in PRELOAD_MODELS}
#%%
import json
prior_counts = np.array(json.load(open(paths.models / 'star_counts.json')))

#%%
class LMClassifier:
    def __init__(self, models, prior_counts, sentiment_weights=[-1, -1, 0, 1, 1.]):#[-1, -.5, 0, .5, 1]):
        self.models = models
        self.prior_logprobs = np.log(prior_counts / prior_counts.sum())
        self.sentiment_weights = np.array(sentiment_weights)
        self.sentiment_weights -= np.min(self.sentiment_weights)
        self.sentiment_weights /= np.max(self.sentiment_weights)

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
        return new_state, score_deltas

    def classify_seq(self, state, toks):
        logprobs = self.prior_logprobs.copy()
        for tok in toks:
            state, score_deltas = self.advance_state(state, tok)
            logprobs += score_deltas
        logprobs -= logsumexp(logprobs)
        return np.exp(logprobs)

    def sentiment(self, state, toks):
        probs = self.classify_seq(state, toks)
        return probs @ self.sentiment_weights



lmc = LMClassifier([models[f'yelp_train-{star}star'] for star in range(1,6)], prior_counts)
for phr in ['this place was terrible', 'this place was amazing', 'this place was reasonably', 'my favorite', 'the only redeeming', 'i wanted to', 'we came here', 'service was slow', 'the service was very friendly']:
    state = lmc.get_state(['<D>'], bos=True)
    hist = []
    seq = phr.split()
    for i in range(len(seq) + 1):
        hist.append(lmc.classify_seq(state, seq[:i]))
    final = hist[-1]
    score = sum(final[-2:]) - sum(final[:2])
    score = (score / 2) + .5
    print(f'{score:.2f} {phr}')
    print(np.round(np.array(hist).T, 2))
    print(f'{lmc.sentiment(state, seq):.2}')
    print()
#    for tok in seq.split():
#        state = lmc.advance_state(state, tok)
#        hist.append(lmc.eval_posterior(state))
#    print(np.round(hist, 2))
