# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:13:01 2017

@author: kcarnold
"""
import hashlib
import random
import pickle
import numpy as np
import pandas as pd
#%%

log_data, survey_data = pickle.load(open('data/analysis_study4_2017-04-02T17:14:44.194603.pkl', 'rb'))
participants = sorted(log_data.keys())

def split_randomly_without_overlap(remaining_views, chunk_size, rs):
    chunks = []
    while sum(remaining_views) >= chunk_size:
        chunk = []
        for i in range(chunk_size):
            mrv = max(remaining_views)
            opts = [i for i, rv in enumerate(remaining_views) if rv == mrv and i not in chunk]
#            item = np.argmax(remaining_views)
            item = rs.choice(opts)
            assert item not in chunk
            chunk.append(item)
            remaining_views[item] -= 1
        chunks.append(chunk)
    return chunks
#split_randomly_without_overlap([1]*10, 4, rs=random.Random(0))

if False:
    CHUNK_SIZE = 4
    VIEWS_PER_ITEM = 3
    splits = split_randomly_without_overlap(len(participants), CHUNK_SIZE, VIEWS_PER_ITEM, rs=random.Random(0))
    data = [{
            "pages": [[
                    dict(participant_id=participants[idx], cond=block['condition'], text=block['finalText']) for block in all_log_analyses[participants[idx]]['blocks']]
                    for idx in chunk],
            "attrs": ["food", "drinks", "atmosphere", "service", "value"],
            } for chunk in splits]

    pd.DataFrame(dict(data=[json.dumps(d) for d in data])).to_csv(f'analyzed_{"_".join(participants)}.csv', index=False)


#%%

#    to_rate = [['852f7a', '88d3ad', '0cb74f', 'f31d92'], ['4edc26', '885dae', 'a997ed', '8c01ef'], ['773fa0', '43cd2c', '706d74', '7d5d97']]
    to_rate = [participants]#list(cytoolz.partition(4, non_excluded_participants))
    data = [{
            "pages": [[
                    dict(participant_id=participant_id, cond=block['condition'], text=block['finalText']) for block in all_log_analyses[participant_id]['blocks']]
                    for participant_id in chunk],
            "attrs": ["food", "drinks", "atmosphere", "service", "value"],
            } for chunk in to_rate]
#%%
    pd.DataFrame(dict(data=[json.dumps(d) for d in data])).to_csv(f'to_rate_{run_id}.csv', index=False)
#%%


# Dump  for spreadsheet raters
def should_flip(participant_id):
    return np.random.RandomState(
            np.frombuffer(hashlib.sha256(participant_id.encode('utf8')).digest(), dtype=np.uint32)
            ).rand() < .5

def make_participant_hash(participant_id):
    return hashlib.sha256(participant_id.encode('utf8')).hexdigest()[:4]
#%%
rate_round_1 = ['8ddf8b', '6a8a4c', '8e4d93', 'a178d3']
rate_round_2 =['10317e',  '3822a7',  '42a2d1',  '51aa50', '60577e', '72b6f6', '83ada3', '993876', 'aae8e4', 'ec0620']

import contextlib

def dump_rating_task(basename, participants, log_data):
    participant_hashes = []

    with open(f'{basename}-reviews.txt', 'w') as f, contextlib.redirect_stdout(f):
        for participant_id in participants:
            texts = [block['finalText'] for block in log_data[participant_id]['blocks']]
            if should_flip(participant_id):
                texts = texts[::-1]
            participant_hash = make_participant_hash(participant_id)
            participant_hashes.append(participant_hash)
            print()
            print(participant_hash)
            print('----')
            for i, text in enumerate(texts):
#                text_hash = hashlib.sha256(text.encode('utf8')).hexdigest()[:2]
                name_str = 'AB'[i]
                print(f"{participant_hash}-{name_str}")
                print(text.replace('\n',' '))
                print()

    with open(f'{basename}-results.csv', 'w') as f, contextlib.redirect_stdout(f):
        for participant_hash in participant_hashes:
            for attr in ["food", "drinks", "atmosphere", "service", "value", "detailed", "written", "quality"]:
                print(f"{participant_hash},{attr},,")

#dump_rating_task('data/detail_ratings/input batches/round2', rate_round_2, log_data)
#%%
conditions = []
for author_id in participants:
    conds = log_data[author_id]['conditions']
    if should_flip(author_id):
        conds = conds[::-1]
    conditions.append([author_id, conds[0], conds[1], ','.join(conds)])
conditions_as_rated = pd.DataFrame(conditions, columns=['author_id', 'cond_A', 'cond_B', 'conds'])
#%% Analyze the dumped files
participant_hash2id = pd.Series(participants, index=[make_participant_hash(x) for x in participants])
rater_ids = ['kf', 'km']
results = pd.concat([
        pd.concat([
                pd.read_csv(f'data/detail_ratings/batch{i+1}_{rater}.csv', header=None, names=['hash', 'attr', 'comparison', 'score_A', 'score_B'])
                for i in range(1)], ignore_index=True)
        for rater in rater_ids],
        keys=rater_ids, names=['rater']).reset_index(level=0).reset_index(drop=True)
results['comparison'] = results.comparison.str.lower()
results['comp'] = results.comparison.map(lambda x: ['a', 'same', 'b'].index(x) - 1)
results = pd.merge(results, participant_hash2id.to_frame('author_id'), left_on='hash', right_index=True, how='left')
results['item_code'] = results.author_id.str.cat(results.attr, sep='-')
results = pd.merge(results, conditions_as_rated, left_on='author_id', right_on='author_id', how='left')
#%%
from nltk.metrics.agreement import AnnotationTask
def interval_distance(a, b):
#    return abs(a-b)
    return pow(a-b, 2)
{col: AnnotationTask(data=results.groupby(('rater', 'author_id')).mean().reset_index().loc[:, ['rater', 'author_id', col]].values.tolist(), distance=interval_distance).alpha()
 for col in ['comp', 'score_A', 'score_B']}
