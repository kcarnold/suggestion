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
a, b = sa.search_range(('<S>', ''))
state = model.bos_state
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
scores = np.array([model.score_seq(state, k)[0] for k in tqdm.tqdm(starts_keys)])
#%%
sent_start_indices = np.array([[model.model.vocab_index(w) for w in sent] for sent in starts_keys])
#%%
any_unk = np.any(sent_start_indices == 0, axis=1)#[any(idx == 0 for idx in sent_indices) for sent_indices in sent_start_indices]
#%%
unigram_probs = model.unigram_probs
unigram_llks_for_start = np.min(unigram_probs[sent_start_indices], axis=1)
unigram_llks_for_start[any_unk] = 0
#%%
#def show_ends(array, names):
#    argsort = np.argsort(array)
#    print()
import pandas as pd
df = pd.DataFrame(dict(start=starts_keys_join, unigram_llk=unigram_llks_for_start, scores=scores))
df['acceptability'] = df.scores - starts_char_lens*df.unigram_llk / 5
df['acnear0'] = -df.acceptability.abs()
#acceptability = scores - unigram_llks_for_start
#starts_keys[np.argmax(acceptability)]
#acceptability
#df.query('unigram_llk < 0').sort_values('acnear0').tail(20)
df.query('unigram_llk < 0').sort_values('acceptability').start.tail(20)