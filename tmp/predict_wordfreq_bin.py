# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:47:53 2017

@author: kcarnold
"""

import numpy as np
from suggestion import suggestion_generator
from scipy.special import logsumexp
from scipy.stats import rankdata
#%%
model = suggestion_generator.get_model('yelp_train-balanced')
#%%
wf_bins_rank = rankdata(model.unigram_probs, method='average')
# np.arange(len(model.unigram_probs_wordsonly))[
#wf_bins_rank = np.argsort(model.unigram_probs_wordsonly)
wf_bins = (10 * wf_bins_rank / (wf_bins_rank.max() + 1)).astype(int)
bin_counts = np.bincount(wf_bins)
#%%
for word in 'huevos tri place'.split():
    idx = model.model.vocab_index(word)
    print(f"{model.unigram_probs_wordsonly[idx]:.2f}, bin={wf_bins[idx]}")

#%%
mean_probs = model.unigram_probs_wordsonly @ np.eye(10)[wf_bins]
#%%
# bin 6 seems high, and bin 1. Why?
[model.id2str[idx] for idx in np.flatnonzero(wf_bins == 6)[:20]]
[wf_bins_rank[idx]/wf_bins_rank.max() for idx in np.flatnonzero(wf_bins == 0)[:20]]
#model.unigram_probs[]
#%%
next_words = np.flatnonzero(~model.is_special)
for sofar in ['', 'best', 'i', 'i really', 'i love their', 'i love their vegan huevos', 'i love their turkey', 'this']:
    sofar = sofar + ' '
    toks = suggestion_generator.tokenize_sofar(sofar)#'<D> best breakfast menu of all places such as'.split()

    state = model.get_state(toks, bos=True)[0]
#    next_words, logprobs = model.next_word_logprobs_raw(state, toks[-1])
    logprobs = model.eval_logprobs_for_words(state, next_words)

    #logprobs[logprobs < np.percentile(logprobs, 75)] = -np.
    bins_for_next_words = wf_bins[next_words]
    logprobs += 100
    bin_probs = logprobs @ np.eye(10)[bins_for_next_words]
    # ^^ this line is suspect.
    logprobs -= 100
#    bin_probs -= mean_probs
    bin_probs -= logsumexp(bin_probs)
    chosen_bin = np.argmax(bin_probs)
    indices_in_bin = np.flatnonzero(bins_for_next_words == chosen_bin)
    words_in_bin = [model.id2str[next_words[idx]] for idx in indices_in_bin]
    normal_indices = np.argsort(logprobs)[-3:]
    chosen_word_normal = [model.id2str[next_words[idx]] for idx in normal_indices]
    chosen_word_binned = [words_in_bin[idx] for idx in np.argsort(logprobs[indices_in_bin])[-3:]]
    print(f"[{sofar}]")
    print("Normal:", ', '.join(chosen_word_normal), [wf_bins[next_words[idx]] for idx in normal_indices])
    print(f"Binned ({chosen_bin}):", ', '.join(chosen_word_binned))
    print("Bin probs:", np.round(bin_probs, 1).tolist())
    print()
