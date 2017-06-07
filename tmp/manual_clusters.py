# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:04:01 2017

@author: kcarnold
"""
import re
import string
import numpy as np

with open('clusters.txt') as f:
    groups = []
    group = []
    meta = None
    for line in f:
        line = line.strip()
        if line.startswith('='):
            meta = line[1:].strip()
            continue
        if not line:
            if group:
                groups.append((meta, group))
                group = []
            continue
        if line[0] in string.digits:
            continue
        group.append(line)
    if group:
        groups.append((meta, group))
#%%
from suggestion import suggestion_generator
from scipy.special import expit
clf = suggestion_generator.CLASSIFIERS['positive']
#%%
from suggestion.lang_model import LMClassifier
clf = LMClassifier([
    suggestion_generator.get_model(f'yelp_train-{star}star') for star in [1, 2, 4, 5]], [-.5, -.5, .5, .5])
clf.classify_seq(clf.get_state([]), "i wouldn't recommend this place".split())
#%%
done_indices = []
for i in range(10):
    while True:
        group_idx = np.random.choice(len(groups))
        if group_idx in done_indices:
            continue
        meta, group = groups[group_idx]
        if i == 0 and meta not in ['START', 'EARLY']:
            continue
        if meta == "EARLY" and i > 1:
            continue
        if i == 9 and meta != "END":
            continue
        done_indices.append(group_idx)
        break
    sent = np.random.choice(group)
    positivity = clf.classify_seq(clf.get_state([]), sent.split())
    print(f'{positivity:.2f} {sent}')
#%%
seq = "it's a little pricy for".split()
for star in [1,2,4,5]:
    model = suggestion_generator.get_model(f'yelp_train-{star}star')
    print(star, model.score_seq_by_word(model.bos_state, seq))
# Turns out 'pricy' only occurs 10 times in 1-star reviews, 48 times in 2-star, 209 times in 4-star, 68 times in 5-star
