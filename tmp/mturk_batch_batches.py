# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:43:07 2017

@author: kcarnold
"""

# Batch MTurk batches
# Created with the help of the following on the MTurk Manage screen:
# Array.from(document.querySelectorAll('a[id*="batch_status"]')).forEach(x => {let f = document.createElement('iframe'); f.setAttribute('width', '600px'); f.setAttribute('height', '600px'); f.setAttribute('src', x.getAttribute('href')+'/download'); document.body.appendChild(f);})
# or just
# batches.forEach(batch => { let f = document.createElement('iframe'); f.setAttribute('width', '600px'); f.setAttribute('height', '600px'); f.setAttribute('src', `https://requester.mturk.com/batches/${batch}/download`); document.body.appendChild(f);})


#%%
import pandas as pd
import glob
#%%
csvs = sorted(glob.glob('*.csv'))
dfs = [pd.read_csv(csv) for csv in csvs]
#%%
full_concat = pd.concat(dfs, axis=0).drop_duplicates(subset='AssignmentId', keep='first')
concats = pd.concat(dfs, axis=0, join='inner').drop_duplicates(subset='AssignmentId', keep='first')
other_axis = pd.Index(concats.columns.tolist() + ['Answer.code'])
concats = pd.concat(dfs, axis=0, join_axes=[other_axis]).drop_duplicates('AssignmentId', keep='first')
concats.to_csv('all_assignments.csv', index=False)


# You'll also find this helpful:
# copy(Array.from(document.querySelectorAll('#batches_reviewable a[id*="batch_status"]')).map(x => (`${x.textContent},${x.getAttribute('href').slice('/batches/'.length)}`)).join('\n'))
# copy(Array.from(document.querySelectorAll('#batches_reviewed a[id*="batch_status"]')).map(x => (`${x.textContent},${x.getAttribute('href').slice('/batches/'.length)}`)).join('\n'))
