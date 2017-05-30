# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:17:29 2017

@author: kcarnold
"""

import json
import arrow
import numpy as np
#%%
with open('logs/d85560.jsonl') as f:
    entries = [json.loads(line) for line in f]
#%%
deltas = np.array([(arrow.get(entry['jsTimestamp']/1000) - arrow.get(entry['timestamp'])).total_seconds() for entry in entries])
deltas -= np.mean(deltas)
#%%
types = [entry['type'] for entry in entries]
ext_acts = np.flatnonzero(np.array(types) == 'externalAction')
#%%
from collections import Counter
Counter([entries[x-1]['type'] for x in np.argsort(np.abs(deltas))[-100:]]).most_common()
#%%
Counter([entry['type'] for entry in entries[100:400]]).most_common()
