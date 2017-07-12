# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:16:03 2017

@author: kcarnold
"""

from suggestion import suggestion_generator
from scipy.special import entr
from scipy.misc import logsumexp
import numpy as np
model = suggestion_generator.get_model('yelp_train-balanced')
#%%
def get_entropy_by_word(seq):
    entropies = []
    lses = []
    for i in range(1, len(seq)):
        x = seq[:i]
        state = model.get_state(x, bos=False)[0]
        indices, logprobs = model.next_word_logprobs_raw(state, x[-1])
        lse = logsumexp(logprobs)
        lses.append(lse)
        logprobs -= lse
        probs = np.exp(logprobs)
        entropies.append(entr(probs).sum())
        print([model.id2str[indices[idx]] for idx in np.argsort(logprobs)[-3:]], x)
    return np.array(entropies), np.array(lses)

seq = '<s> <D> i really like this place , but </S>'.split()
entropies, lses = get_entropy_by_word(seq)
list(zip(seq[1:], entropies))
