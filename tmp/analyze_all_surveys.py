# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:22:42 2017

@author: kcarnold
"""

import numpy as np
import pandas as pd
from suggestion.paths import paths
import re
#%%
data_files = list((paths.parent / 'data' / 'surveys').glob('surveys_*.csv'))
latest = {}
for filename in data_files:
    study, date = re.match(r'surveys_(.+)_(2017.+)', filename.name).groups()
    if study not in latest or date > latest[study][0]:
        latest[study] = date, filename
#%%
all_data = pd.concat({study: pd.read_csv(filename) for study, (date, filename) in latest.items()})
all_data.index.names = ['study', None]
all_data = all_data.reset_index('study').reset_index(drop=True)
all_data = all_data.drop_duplicates(['participant_id', 'condition', 'idx', 'name', 'survey', 'value'], keep='last')
#%%
prefix_subs = {
        "How much do you agree with the following statements about the words or phrases that the keyboard...-They ": "suggs-"
        }
def prefix_sub(k):
    for x, y in prefix_subs.items():
        if k.startswith(x):
            k = k.replace(x, y, 1)
            break
    return k

all_data['name'] = all_data.name.apply(prefix_sub)
#%%
def extract_weighted_survey_results(all_survey_data, survey, items):
    return (all_survey_data[all_survey_data.survey == survey]
            .set_index(['participant_id', 'condition', 'idx', 'name'])
            .value.unstack(level=-1)
            .apply(lambda x: sum(weight*x[col] for weight, col in items), axis=1))

opinions_suggs_postTask = [(1, 'suggs-were interesting'),
    (1, 'suggs-were relevant'),
    (-1, 'suggs-were distracting'),
    (1, 'suggs-were helpful'),
    (1, "suggs-made me say something I didn't want to say."),
    (-1, 'suggs-felt pushy.'),
    (1, 'suggs-were inspiring.'),
    (1, 'suggs-were timely.')]
#%%
sug_opins = all_data[all_data.name.str.startswith('suggs-')].copy()
sug_opins['value'] = pd.to_numeric(sug_opins.value)
#%%
sug_opins.to_csv('all_sugg_opinions.csv', index=False)