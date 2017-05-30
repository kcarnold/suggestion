# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:43:07 2017

@author: kcarnold
"""

# Batch MTurk batches
# Created with the help of the following on the MTurk Manage screen:
# Array.from(document.querySelectorAll('a[id*="batch_status"]')).forEach(x => {let f = document.createElement('iframe'); f.setAttribute('width', '600px'); f.setAttribute('height', '600px'); f.setAttribute('src', x.getAttribute('href')+'/download'); document.body.appendChild(f);})

#%%
import pandas as pd
import glob
#%%
csvs = sorted(glob.glob('*.csv'))
dfs = [pd.read_csv(csv) for csv in csvs]
common_area = [df.iloc[:, :27] for df in dfs]
#%%
concats = pd.concat(dfs, axis=0, join='inner').drop_duplicates(subset='AssignmentId', keep='first')
concats.to_csv('all_assignments.csv', index=False)


# You'll also find this helpful:
# copy(Array.from(document.querySelectorAll('#batches_reviewable a[id*="batch_status"]')).map(x => (`${x.getAttribute('href')},${x.textContent}`)).join('\n'))
# copy(Array.from(document.querySelectorAll('#batches_reviewed a[id*="batch_status"]')).map(x => (`${x.getAttribute('href')},${x.textContent}`)).join('\n'))
# And throw that into SublimeText :)
