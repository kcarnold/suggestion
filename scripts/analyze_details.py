# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:43:43 2017

@author: kcarnold
"""

import pandas as pd
import numpy as np
import json
#%%
data_files = ['data/study3 details batch1 Batch_2748636_batch_results.csv']
#data_files = ['data/initial detail ratings - Batch_2741651_batch_results.csv']
#data_files = ['data/full arnold16 details Batch_2742001_batch_results.csv']
#data_files = ['data/detail rating by assignment Batch_2744268_batch_results.csv', 'data/detail rating study2 batch2 Batch_2745786_batch_results.csv']
raw_data = pd.concat([pd.read_csv(f) for f in data_files], axis=0, ignore_index=True)
#%%
def load_json(df, cols):
    df = df.copy()
    df[cols] = df[cols].applymap(json.loads)
    return df
data = (raw_data
#        [raw_data.AssignmentStatus == "Submitted"]
        .pipe(load_json, ['Input.data', 'Answer.results']))

#%%
io_tuples = [(row['WorkerId'], row['Input.data'], row['Answer.results']) for idx, row in data.iterrows()]

#%%
if False:
    io_files = [
            ('me', 'data/my_input_data_20170328.json', 'data/my_results_20170328.json'),
            ('me', 'data/pilot_input_data-2017-03-28.json', 'data/pilot_output_data_20170328.json')]
    io_tuples = [(worker, json.load(open(prompt)), json.load(open(result))) for worker, prompt, result in io_files]

#%%
all_results = []
comparisons = []
num_highlights = []
decode_side = dict(A=0, B=1, neither=None, same=None)
for worker_id, prompt, results in io_tuples:
    all_results.append(dict(worker_id=worker_id, results=results))
    highlights = results['highlights']
    ratings = {}
    attrs = set()
    for key, rating in results['ratings'].items():
        attr, page_num = key.split('-', 1)
        rating = results['ratings'][f'{attr}-{i}']
        favored_side = decode_side[rating]
        ratings[attr, int(page_num)] = favored_side
        attrs.add(attr)


    for i, page in enumerate(prompt['pages']):
        a_cond = page[0]['cond']
        b_cond = page[1]['cond']
        author_id = page[0]['participant_id']
        assert author_id == page[1]['participant_id']
        conds = [a_cond, b_cond]
        for attr in attrs:
            favored_side = ratings.get((attr, i))
            favored_cond = conds[favored_side] if favored_side is not None else None
            comparisons.append(dict(
                    worker_id=worker_id,
                    attr=attr,
                    page_num=i,
                    author_id=author_id,
                    item_code=f'{author_id}-{attr}',
                    favored_side=favored_side,
                    favored_cond=favored_cond))
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
#(comparisons.groupby('worker_id').favored_cond.value_counts() / comparisons.groupby('worker_id').favored_cond.count()).groupby(level=-1).mean()
#%%
num_highlights.groupby('worker_id').num_highlights.mean()
#%%
comparisons.query('attr=="detailed"').favored_cond.value_counts()
#%%
comparisons.favored_cond.value_counts()
#%%
overall_dists = comparisons.query('attr=="overall"').groupby(('author_id')).favored_cond.value_counts(normalize=True).unstack().fillna(0)
overall_dists[overall_dists.max(axis=1) >= .75].mean(axis=0)
#%%
#comparisons.to_csv('data/full_arnold16_details.csv')
#%%

#json.dump(all_results, open('all_detail_results_2017-03-28.json','w'))
#%% Inter-annotator agreement (Krippendorrf's alpha)
from nltk.metrics.agreement import AnnotationTask
base_alpha = AnnotationTask(data=comparisons.loc[:, ['worker_id', 'item_code', 'favored_cond']].values.tolist()).alpha()
alpha_without = {worker_id: AnnotationTask(data=comparisons[comparisons.worker_id != worker_id].loc[:, ['worker_id', 'item_code', 'favored_cond']].values.tolist()).alpha() for worker_id in comparisons.worker_id.unique()}
#%%
num_highlights_by_worker = num_highlights=num_highlights.groupby('worker_id').num_highlights.sum()
pd.DataFrame(dict(num_highlights=num_highlights_by_worker, alpha_without=alpha_without)).sort_values('alpha_without')
#sorted(alpha_without.items(), key=lambda x: x[1])

#%%
filtered = pd.merge(comparisons, num_highlights_by_worker.to_frame('num_highlights'), left_on='worker_id', right_index=True)
filtered = filtered[filtered['num_highlights'] > 20]
#%%
base_alpha = AnnotationTask(data=filtered.loc[:, ['worker_id', 'item_code', 'favored_cond']].values.tolist()).alpha()
alpha_without = {worker_id: AnnotationTask(data=filtered[filtered.worker_id != worker_id].loc[:, ['worker_id', 'item_code', 'favored_cond']].values.tolist()).alpha() for worker_id in filtered.worker_id.unique()}
pd.DataFrame(dict(num_highlights=num_highlights_by_worker, alpha_without=alpha_without)).sort_values('alpha_without')
#filtered = comparisons[comparisons.worker_id != 'A3SGZ8OQXW5TQD']
#filtered.query('attr=="overall"').favored_cond.value_counts()
#%%
filtered.favored_cond.value_counts()
