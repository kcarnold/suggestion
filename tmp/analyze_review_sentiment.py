# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:18:57 2017

@author: kcarnold
"""

import pandas as pd
import json
import numpy as np
#%%
PARTICIPANT_LEVEL_CSV = 'data/by_participant/participant_level_sent3_2_2017-06-21T08:52:31.507194.csv'
ANNOTATIONS_JSON = 'sent3_2-sentiment-results.json'
#%%
data = pd.read_csv(PARTICIPANT_LEVEL_CSV).query('kind == "final"')
results = json.load(open(ANNOTATIONS_JSON))['allStates']
#%%
pos_neg = pd.DataFrame([
        dict(text=ent['text'],
             pos=sum(x['range'][1] - x['range'][0] for x in ent['annotations'] if x['tool'] == 'pos') or 0.,
             neg=sum(x['range'][1] - x['range'][0] for x in ent['annotations'] if x['tool'] == 'neg') or 0.)
        for ent in results])
#%%
with_pos_neg = pd.merge(data, pos_neg, left_on='finalText', right_on='text')
#%%
pos = with_pos_neg['pos']
neg = with_pos_neg['neg']
with_pos_neg['diversity'] = 1 - (np.abs(pos - neg) / (pos + neg))
#%%
with_pos_neg.to_csv('annotated_sent3_2-sentiment-results.csv')
#%%
data['pos'] = pos_neg.pos
data['neg'] = pos_neg.neg
#pd.concat([data, pos_neg], axis=1)