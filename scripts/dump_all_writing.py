# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:45:44 2017

@author: kcarnold
"""

import pathlib
import pandas as pd
from suggestion.paths import paths
import re
#%%
data_files = list((paths.parent / 'data' / 'by_participant').glob('participant_level_*.csv'))
latest = {}
for filename in data_files:
    study, date = re.match(r'participant_level_(.+)_(2017.+)', filename.name).groups()
    if study not in latest or date > latest[study][0]:
        latest[study] = date, filename
#%%
all_data = pd.concat({study: pd.read_csv(filename) for study, (date, filename) in latest.items()})
#%%
all_data.index.set_names(level=0, names='study_name', inplace=True)
all_data.drop_duplicates('finalText', keep='last').reset_index(level=1, drop=True).reset_index().to_csv('data/all_writing.csv', index=False)