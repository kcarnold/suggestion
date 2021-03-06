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
#data_file = 'data/analysis_study4_2017-04-02T17:14:44.194603.pkl'
#data_file = 'data/analysis_study4_2017-04-02T20:37:11.374099.pkl'
#data_file = 'data/analysis_study4_2017-04-02T21:09:39.528242.pkl'
#data_file = 'data/analysis_study4_2017-04-04T13:11:10.932814.pkl'
data_file = 'data/analysis_funny_2017-04-07T09:58:07.316857.pkl'
log_data, survey_data = pickle.load(open(data_file, 'rb'))
participants = sorted(log_data.keys())
#%%
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
rate_round_3 = ['10f0dc', 'ac1341', 'b2d633', 'c8963d']
rate_round_4 = ['7939c9', '8a8a64', 'bb9486', 'c7ffcb']
rate_round_5 = ['ab938b']
all_rated = set(rate_round_1 + rate_round_2 + rate_round_3 + rate_round_4 + rate_round_5)
#%%
sorted(set(participants) - set(all_rated))
#%%

texts = {pid: [block['finalText'] for block in data['blocks']] for pid, data in log_data.items()}
#texts = {pid: [block['condition'] for block in data['blocks']] for pid, data in log_data.items()}
#%%
import contextlib

def dump_rating_task(basename, participants, texts_by_participant_id):
    participant_hashes = []

    with open(f'{basename}-reviews.txt', 'w') as f, contextlib.redirect_stdout(f):
        for participant_id in participants:
            texts = texts_by_participant_id[participant_id]
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
#%%
dump_rating_task('data/detail_ratings/input batches/funny1', list(participants), texts)
#%%
participants = sorted(list(log_data.keys()))
conditions = []
for author_id in participants:
    author_conds = log_data[author_id]['conditions']
    if should_flip(author_id):
        rating_conds = author_conds[::-1]
    else:
        rating_conds = author_conds
    conditions.append([author_id, rating_conds[0], rating_conds[1], ','.join(author_conds)])
conditions_as_rated = pd.DataFrame(conditions, columns=['author_id', 'cond_A', 'cond_B', 'author_conds'])
#%% Analyze the dumped files
participant_hash2id = pd.Series(participants, index=[make_participant_hash(x) for x in participants])

ratings_files = [f'batch{batch+1}_{rater}' for batch in range(4) for rater in ['kf', 'km']]
#ratings_files = [f'old_{rater}' for rater in ['kf', 'km']]

#ratings_files = ['test']
results = pd.concat([
        pd.read_csv(f'data/detail_ratings/{fname}.csv', header=None, names=['hash', 'attr', 'comparison', 'score_A', 'score_B'])
            .assign(rater=fname.split('_')[-1])
        for fname in ratings_files], ignore_index=True)
results['comparison'] = results.comparison.str.lower()
results['comp'] = results.comparison.map(lambda x: ['a', 'same', 'b'].index(x) - 1)
#for col in ['score_A', 'score_B']: #'comp',
#    results[col] = results.groupby('rater')[col].transform(lambda x: (x-x.mean()) / x.std())
results = pd.merge(results, participant_hash2id.to_frame('author_id'), left_on='hash', right_index=True, how='left')
results['item_code'] = results.author_id.str.cat(results.attr, sep='-')
results = pd.merge(results, conditions_as_rated, left_on='author_id', right_on='author_id', how='left')
results
#%%
def in_author_order(row):
    '''Convert the rater row to author order, and in numeric form.

    result:
        - comparison: -1 means author's first wins, 0=same, 1=second one wins.
        - score_first
        - score_second
    '''
    result = row.copy()
    flip = should_flip(row['author_id'])
    # Encode the binary comparison.
    comparison = row['comp']
    if flip:
        comparison = -comparison
    result['comparison_author'] = comparison
    scores = row['score_A'], row['score_B']
    if flip:
        scores = scores[::-1]
    result['score_first'], result['score_second'] = scores
    return result
results2 = results.apply(in_author_order, axis=1)
results2.to_csv('data/detail_ratings.csv')
#%%
standardized = results2.copy()
for col in ['comparison_author', 'score_A', 'score_B', 'score_first', 'score_second']:
    standardized[col] = standardized.groupby('rater')[col].transform(lambda x: (x-x.mean()) / x.std())
standardized.to_csv('data/detail_ratings_standardize_complete_study4.csv')
#%%
from nltk.metrics.agreement import AnnotationTask
def interval_distance(a, b):
#    return abs(a-b)
    return pow(a-b, 2)
{col: AnnotationTask(data=results2.groupby(('rater', 'author_id')).mean().reset_index().loc[:, ['rater', 'author_id', col]].values.tolist(), distance=interval_distance).alpha()
 for col in ['comp', 'score_A', 'score_B']}



#%% Dump data to analyze arnold16 ratings.
import json
arnold16 = pd.read_csv('data/arnold16_full_participant_data.csv')
arnold16_filtered = arnold16[arnold16.idx >= 2]
arnold16_grouped_by_participant = {pid: [data[f'{i}'] for i in range(2,4)] for pid, data in
                                         json.loads(arnold16_filtered.set_index(['participant_id', 'idx']).reviewText.unstack().to_json(orient='index')).items()}
#for participant_id, texts in grouped_by_participant.items():
#    order = rs.sample('pw', 2)
#    pairs.append([(dict(participant_id=participant_id, cond=cond, text=texts[cond])) for cond in order])
dump_rating_task('data/detail_ratings/input batches/old', sorted(arnold16_grouped_by_participant.keys()), arnold16_grouped_by_participant)



#%% Analyze the results
participants = sorted(arnold16_grouped_by_participant.keys())
author_conditions = {str(k): [x[2], x[3]] for k, x in arnold16_filtered.set_index(['participant_id', 'idx']).condition.unstack(-1).to_dict('index').items()}
conditions = []
for author_id in participants:
    author_conds = author_conditions[author_id]
    if should_flip(author_id):
        rating_conds = author_conds[::-1]
    else:
        rating_conds = author_conds
    conditions.append([author_id, rating_conds[0], rating_conds[1], ','.join(author_conds)])
conditions_as_rated = pd.DataFrame(conditions, columns=['author_id', 'cond_A', 'cond_B', 'author_conds'])

participant_hash2id = pd.Series(participants, index=[make_participant_hash(x) for x in participants])
ratings_files = [f'old_{rater}' for rater in ['kf', 'km']]
results = pd.concat([
        pd.read_csv(f'data/detail_ratings/{fname}.csv', header=None, names=['hash', 'attr', 'comparison', 'score_A', 'score_B'])
            .assign(rater=fname.split('_')[-1])
        for fname in ratings_files], ignore_index=True)
results['comparison'] = results.comparison.str.lower()
results['comp'] = results.comparison.map(lambda x: ['a', 'same', 'b'].index(x) - 1)
results = pd.merge(results, participant_hash2id.to_frame('author_id'), left_on='hash', right_index=True, how='left')
results['item_code'] = results.author_id.str.cat(results.attr, sep='-')
results = pd.merge(results, conditions_as_rated, left_on='author_id', right_on='author_id', how='left')
for col in ['comp', 'score_A', 'score_B']:
    results[col] = results.groupby('rater')[col].transform(lambda x: (x-x.mean()) / x.std())
results = results.apply(in_author_order, axis=1)

results['score_diff'] = results['score_second'] - results['score_first']
results.to_csv('data/arnold16_details.csv')
#%%
from nltk.metrics.agreement import AnnotationTask
def interval_distance(a, b):
#    return abs(a-b)
    return pow(a-b, 2)
{col: AnnotationTask(data=results.groupby(('rater', 'author_id')).mean().reset_index().loc[:, ['rater', 'author_id', col]].values.tolist(), distance=interval_distance).alpha()
 for col in ['comparison_author', 'score_first', 'score_second', 'score_diff']}


#.to_csv('arnold16_details.csv')

