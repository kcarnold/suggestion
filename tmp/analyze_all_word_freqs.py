# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:39:40 2017

@author: kcarnold
"""

import numpy as np
import pandas as pd
from suggestion.paths import paths
import re
#%%
data_files = list((paths.parent / 'data' / 'by_participant').glob('participant_level_*.csv'))
latest = {}
for filename in data_files:
    study, date = re.match(r'participant_level_(.+)_(2017.+)', filename.name).groups()
    if study not in latest or date > latest[study][0]:
        latest[study] = date, filename
#%%
all_data = pd.concat({study: pd.read_csv(filename) for study, (date, filename) in latest.items()})
all_data.index.names = ['study', None]
all_data = all_data.reset_index('study').reset_index(drop=True)
all_data = all_data.drop_duplicates(['participant_id', 'condition', 'block', 'kind'])
#%%
all_data[all_data.study == 'study4'].kind.value_counts()
#%%
from suggestion.analyzers import WordFreqAnalyzer
analyzer = WordFreqAnalyzer.build()
#%%
# Hapax legomena are pruned by KenLM. Should we prune more?
[analyzer.vocab[i] for i in np.flatnonzero(analyzer.counts == 5)[:5]]
#%% Ok let's ignore any word with count < 5 in Yelp.
from suggestion import tokenization
import string
def analyze(doc):
    toks = tokenization.tokenize(doc)[0]
    filtered = []
    freqs = []
    for tok in toks:
        if tok[0] not in string.ascii_letters:
            continue
        vocab_idx = analyzer.word2idx.get(tok)
        if vocab_idx is None or analyzer.counts[vocab_idx] < 5:
            print("Skipping", tok)
            continue
        filtered.append(tok)
        freqs.append(analyzer.log_freqs[vocab_idx])
    return pd.Series(dict(wf_N=len(freqs), wf_mean=np.mean(freqs), wf_std=np.std(freqs)))
word_freq_data = all_data.query('kind == "final"').finalText.apply(analyze)
with_word_freq = pd.merge(all_data, word_freq_data, left_index=True, right_index=True)
#%%
with_word_freq.to_csv('all_word_freqs.csv', index=False)