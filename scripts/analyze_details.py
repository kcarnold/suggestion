# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:43:43 2017

@author: kcarnold
"""

import pandas as pd
import numpy as np
import json
#%%
data = pd.read_csv('data/initial detail ratings - Batch_2741651_batch_results.csv')
#%%
prompts = [json.loads(x) for x in data['Input.data']]
results = [json.loads(x) for x in data['Answer.results']]
#%%
comparisons = []
num_highlights = []
decode_side = dict(A=0, B=1, neither=None)
for idx, row in data.iterrows():
    worker_id = row['WorkerId']
    prompt = json.loads(row['Input.data'])
    results = json.loads(row['Answer.results'])
    highlights = results['highlights']
    attrs = prompt['attrs']
    for i, page in enumerate(prompt['pages']):
        a_cond = page[0]['cond']
        b_cond = page[1]['cond']
        conds = [a_cond, b_cond]
        for attr in attrs:
#            print(worker_id, attr, i, results['ratings'][f'{attr}-{i}'])
            favored_side = decode_side[results['ratings'][f'{attr}-{i}']]
            favored_cond = conds[favored_side] if favored_side is not None else None
            comparisons.append(dict(
                    worker_id=worker_id,
                    attr=attr,
                    page_num=i,
                    favored_side=favored_side,
                    favored_cond=favored_cond))
        for side in range(2):
            num_highlights.append(dict(
                    worker_id=worker_id,
                    page_num=i,
                    side=side,
                    num_highlights=len(highlights[f'{i}-{side}']['annotations'])))
#        print(worker_id, i, len(highlights[f'{i}-0']['annotations']), len(highlights[f'{i}-1']['annotations']))

comparisons = pd.DataFrame(comparisons)
num_highlights = pd.DataFrame(num_highlights)
#%%
comparisons.groupby('worker_id').favored_cond.value_counts()