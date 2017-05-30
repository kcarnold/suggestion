# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:04:56 2017

@author: kcarnold
"""

from suggestion.paths import paths
import pandas as pd
import nltk
from suggestion import suggestion_generator
#%%
all_writing = pd.read_csv(paths.data / 'all_writing.csv')
#%%
bos_sugg_flag = 'continue'
contexts = []
for text in all_writing.finalText:
    sents = nltk.sent_tokenize(text)
    sug_state = None
    suggs_out = []
    for i in range(len(sents)+1):
        sofar = ' '.join(sents[:i])
        suggs, sug_state, meta = suggestion_generator.get_bos_suggs(sofar, sug_state, constraints={}, bos_sugg_flag=bos_sugg_flag)
        if suggs is None:
            continue
        suggs = [' '.join(s) for s, _ in suggs]
        suggs_out.append(dict(sent=sents[i-1] if i else '', suggs=suggs, meta=meta))
    contexts.append(dict(text=text, suggs=suggs_out))
#%%
import json
json.dump(contexts, open('data/bos_sugg_replays.json','w'))
