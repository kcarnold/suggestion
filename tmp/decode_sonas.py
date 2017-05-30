# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 21:01:05 2017

@author: kcarnold
"""
import os.path
import cytoolz
#%%
#%%
groups = cytoolz.groupby(0, sonas)
print('\n'.join(map(','.join, [(sid, max((os.path.getsize(f'logs/{pid}.jsonl'), pid) for sid, pid in pids)[1]) for sid, pids in groups.items()])))