# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:43:43 2017

@author: kcarnold
"""

import pandas as pd
import numpy as np
import json
#%%
#data = pd.read_csv('data/initial detail ratings - Batch_2741651_batch_results.csv')
data = pd.read_csv('data/full arnold16 details Batch_2742001_batch_results.csv')
#%%
def load_json(df, cols):
    df = df.copy()
    df[cols] = df[cols].applymap(json.loads)
    return df
data = data.pipe(load_json, ['Input.data', 'Answer.results'])

#%%
io_tuples = [(row['WorkerId'], row['Input.data'], row['Answer.results']) for idx, row in data.iterrows()]

#%%
io_files = [
        ('me', 'data/my_input_data_20170328.json', 'data/my_results_20170328.json'),
        ('me', 'data/pilot_input_data-2017-03-28.json', 'data/pilot_output_data_20170328.json')]
io_tuples = [(worker, json.load(open(prompt)), json.load(open(result))) for worker, prompt, result in io_files]

#%%
all_results = []
comparisons = []
num_highlights = []
decode_side = dict(A=0, B=1, neither=None)
for worker_id, prompt, results in io_tuples:
    all_results.append(dict(worker_id=worker_id, results=results))
    highlights = results['highlights']
    attrs = prompt['attrs'] + ['written', 'overall']
    for i, page in enumerate(prompt['pages']):
        a_cond = page[0]['cond']
        b_cond = page[1]['cond']
        author_id = page[0]['participant_id']
        assert author_id == page[1]['participant_id']
        conds = [a_cond, b_cond]
        for attr in attrs:
            try:
                rating = results['ratings'][f'{attr}-{i}']
                favored_side = decode_side[rating]
                favored_cond = conds[favored_side] if favored_side is not None else None
                comparisons.append(dict(
                        worker_id=worker_id,
                        attr=attr,
                        page_num=i,
                        author_id=author_id,
                        favored_side=favored_side,
                        favored_cond=favored_cond))
            except KeyError:
                pass
        for side in range(2):
            try:
                num_highlights.append(dict(
                        worker_id=worker_id,
                        page_num=i,
                        side=side,
                        num_highlights=len(highlights[f'{i}-{side}']['annotations'])))
            except KeyError:
                pass
#        print(worker_id, i, len(highlights[f'{i}-0']['annotations']), len(highlights[f'{i}-1']['annotations']))

comparisons = pd.DataFrame(comparisons)
num_highlights = pd.DataFrame(num_highlights)
#%%
(comparisons.groupby('worker_id').favored_cond.value_counts() / comparisons.groupby('worker_id').favored_cond.count()).groupby(level=-1).mean()
#%%
num_highlights.groupby('worker_id').num_highlights.mean()
#%%

json.dump(all_results, open('all_detail_results_2017-03-28.json','w'))