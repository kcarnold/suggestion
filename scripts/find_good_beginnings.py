# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:43:39 2017

@author: kcarnold
"""
from suggestion import suggestion_generator
import itertools
model = suggestion_generator.get_model('yelp_train')
import tqdm
import datrie
import numpy as np
#%%
sa = suggestion_generator.sufarr
a, b = sa.search_range(('<D>', ''))
chars = sorted(set(itertools.chain.from_iterable(model._bigrams[0].values()))) + [' ']
sent_starts = datrie.Trie(''.join(chars))
for i in tqdm.tqdm(range(a,b)):
    sent_starts[' '.join(sa.docs[sa.doc_idx[i]][sa.tok_idx[i]+1:][:5])] = 1
#%%
starts_keys = [k.split() for k in sent_starts.keys()]
#%%
starts_keys = [start for start in starts_keys if len(start) == 5 and '.' not in start and '</S>' not in start]
#%%
starts_keys_join = [' '.join(start) for start in starts_keys]
starts_char_lens = np.array([len(start) for start in starts_keys_join])
#%%
state = model.get_state(['<s>', '<D>'])[0]
scores = np.array([model.score_seq(state, k)[0] for k in tqdm.tqdm(starts_keys)])
#%%
sent_start_indices = np.array([[model.model.vocab_index(w) for w in sent] for sent in starts_keys])
#%%
any_unk = np.any(sent_start_indices == 0, axis=1)#[any(idx == 0 for idx in sent_indices) for sent_indices in sent_start_indices]
#%%
tags = model.pos_tags[sent_start_indices]
#%%
unigram_probs = model.unigram_probs
unigram_llks_for_start = np.mean(unigram_probs[sent_start_indices], axis=1) # * (tags == 1)
unigram_llks_for_start[any_unk] = 0
#%%
#def show_ends(array, names):
#    argsort = np.argsort(array)
#    print()
import pandas as pd
df = pd.DataFrame(dict(start=starts_keys_join, unigram_llk=unigram_llks_for_start, scores=scores, lens=starts_char_lens))
df['acceptability'] = df.scores - 1*df.unigram_llk
#df['acnear0'] = -df.acceptability.abs()
#acceptability = scores - unigram_llks_for_start
#starts_keys[np.argmax(acceptability)]
#acceptability
#df.query('unigram_llk < 0').sort_values('acnear0').tail(20)
df.query('unigram_llk < 0 and lens >= 30').sort_values('acceptability').start.tail(20)
#%%
unigram_probs[sent_start_indices[646728]]
#%%

#%%
# Load in the cluster models defined in cluster_sents.py
# <><><><>
# ok done.

#start_states = [model.bos_state for model in models]
scores_by_cluster = np.array([[model.score_seq(model.bos_state, k)[0] for model in models] for k in tqdm.tqdm(starts_keys)])
#%%
from scipy.misc import logsumexp
sbc_scale = .25 * scores_by_cluster + 1*scores[:,None] - 1 * unigram_llks_for_start[:,None]
likelihood_bias = logsumexp(sbc_scale, axis=1, keepdims=True)
scores_by_cluster_debias = sbc_scale - 1*likelihood_bias - 0*scores[:,None]
most_distinctive = np.argmax(scores_by_cluster_debias, axis=0)
for cluster, sent_idx in enumerate(most_distinctive):
    print('{:4.2f} {}'.format(np.exp(scores_by_cluster_debias[sent_idx, cluster]), ' '.join(starts_keys[sent_idx])))
#print('\n'.join([) for i in most_distinctive]))