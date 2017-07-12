import re
import os
import pathlib
import pickle
import subprocess
import pandas as pd
import json
import numpy as np
import datetime
import itertools

from suggestion.util import mem

from suggestion.analysis_util import get_existing_requests


#%%
def summarize_freetexts(all_survey_data):
    text_lens = all_survey_data.groupby(['survey', 'name']).value.apply(lambda x: x.str.len().max())
    answers_to_summarize = text_lens[text_lens>20]
    for pid, data_to_summarize in pd.merge(all_survey_data, answers_to_summarize.to_frame('mrl').reset_index(), left_on=['survey', 'name'], right_on=['survey', 'name']).groupby('participant_id'):
        print(pid)
        for (survey, idx), this_survey_data in data_to_summarize.groupby(['survey', 'idx']):
            if survey == 'intro':
                continue
            print(survey, idx)
            for row in this_survey_data.itertuples():
                print(row.condition, row.name, '===', row.value)
            print()
        print()
summarize_freetexts(all_survey_data)
#%%
#topic_self_reports = all_survey_data[all_survey_data.name.str.startswith('How much did you say about each topic')]
decoded_self_reports = pd.concat([
    all_survey_data,
    all_survey_data.name.str.extract(r'How much did you say about each topic.*\.\.\.-(?P<topic_name>.+)-Review (?P<review_idx>\d).*', expand=True),
    all_survey_data.name.str.extract(r'Roughly how many lines of each review were\.\.\.-(?P<sentiment>.+)\?-Review (?P<review_idx>\d).*', expand=True),
    ], axis=1)
self_report_sentiment = (
        decoded_self_reports[~decoded_self_reports.sentiment.isnull()].loc[:,['participant_id', 'condition','sentiment','review_idx', 'value']]
        .dropna(axis=1, how='all')
        .fillna({'value': 0}))
#self_report_sentiment['value'] = pd.to_numeric(self_report_sentiment['value'])
self_report_sentiment = pd.to_numeric(self_report_sentiment.set_index(['participant_id', 'condition', 'review_idx', 'sentiment']).value).unstack().reset_index()
#srs_conditions = self_report_sentiment.condition.str.split(',', expand=True)
#srs_conditions.columns = pd.Index([f'c{i}' for i in range(1,4)])

self_report_sentiment = pd.concat([
    self_report_sentiment,
#    srs_conditions
    self_report_sentiment.apply(lambda row: row['condition'].split(',')[int(row['review_idx'])-1], axis=1).to_frame('rated_condition'),
    ], axis=1)
self_report_sentiment['total_sent'] = self_report_sentiment['positive'] + self_report_sentiment['negative']
self_report_sentiment['positive'] = self_report_sentiment['positive'] / self_report_sentiment['total_sent']
self_report_sentiment['negative'] = self_report_sentiment['negative'] / self_report_sentiment['total_sent']
self_report_sentiment['imbalance'] = np.abs(self_report_sentiment['positive'] - self_report_sentiment['negative'])
self_report_sentiment.to_csv(f'data/self_report_sentiment_{batch_code}.csv', index=False)
#%%
self_report_topic = (
        decoded_self_reports[decoded_self_reports.topic_name.notnull()].loc[:, ['participant_id', 'condition', 'review_idx', 'topic_name', 'value']]
        .dropna(axis=1, how='all')
        .fillna({'value': 0}))
self_report_topic = pd.concat([
        self_report_topic,
        self_report_topic.apply(lambda row: row['condition'].split(',')[int(row['review_idx'])-1], axis=1).to_frame('rated_condition'),
    ], axis=1)
self_report_topic['value'] = pd.to_numeric(self_report_topic['value'])
topic_data = self_report_topic.set_index(['participant_id', 'condition', 'review_idx', 'topic_name']).value.unstack()
topic_data = pd.concat([
        topic_data,
        topic_data.apply(lambda row: row.name[1].split(',')[int(row.name[2])-1], axis=1).to_frame('rated_condition'),
        pd.DataFrame(dict(
                breadth=self_report_topic.groupby(['participant_id', 'condition', 'review_idx']).value.apply(lambda x: np.sum(x > 1)),
                mentions=self_report_topic.groupby(['participant_id', 'condition', 'review_idx']).value.apply(lambda x: np.sum(x > 0)),
                depth=self_report_topic.groupby(['participant_id', 'condition', 'review_idx']).value.apply(lambda x: x[x>0].mean())))
        ], axis=1)
topic_data.to_csv(f'data/self_report_topic_{batch_code}.csv')
#%%
def extract_weighted_survey_results(all_survey_data, survey, items):
    return (all_survey_data[all_survey_data.survey == survey]
            .set_index(['participant_id', 'condition', 'idx', 'name'])
            .value.unstack(level=-1)
            .apply(lambda x: sum(weight*x[col] for weight, col in items), axis=1))
#%%
if False:
    #%%
    all_survey_data = pd.DataFrame(all_survey_data)
#%%
    opinions_suggs_postTask = [(1, 'suggs-were interesting'),
        (1, 'suggs-were relevant'),
        (-1, 'suggs-were distracting'),
        (1, 'suggs-were helpful'),
        (1, "suggs-made me say something I didn't want to say."),
        (-1, 'suggs-felt pushy.'),
        (1, 'suggs-were inspiring.'),
        (1, 'suggs-were timely.')]

    opinions_suggs_postFreewrite = [(1, 'suggs-gave me ideas about what to write about.'),
        (-1, 'suggs-made it harder to come up with my own ideas about what to write about.'),
        (1, 'suggs-gave me ideas about words or phrases to use.'),
        (-1, 'suggs-made it harder for me to come up with my own words or phrases.'),
        (1, 'suggs-were interesting.'),
        (-1, 'suggs-were distracting.'),
        (1, 'suggs-were relevant.'),
        (-1, "suggs-didn't make sense."),
        (-1, 'suggs-were repetitive.'),
        (1, 'suggs-were inspiring.')]

    pd.DataFrame(dict(
        opinion_of_suggs_posttask=extract_weighted_survey_results(all_survey_data, 'postTask', opinions_suggs_postTask),
        opinion_of_suggs_postfreewrite=extract_weighted_survey_results(all_survey_data, 'postFreewrite', opinions_suggs_postFreewrite)),
    ).to_csv('data/opinion_of_suggs.csv')
#%%

    extract_weighted_survey_results(all_survey_data, 'postTask', [(1, 'final-I knew what I wanted to say.'),
        (1, 'final-I knew how to say what I wanted to say.'),
        (-1, 'final-I often paused to think of what to say.'),
        (-1, 'final-I often paused to think of how to say what I wanted to say.'),
        (1, 'final-I was able to express myself fluently.'),
        (-1, 'final-I had trouble expressing myself fluently.')]).to_frame('expression').to_csv('data/expression.csv')

