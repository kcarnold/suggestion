# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 18:28:02 2017

@author: kcarnold
"""

from suggestion import manual_bos
import re
#%%
def fixup(s):
    s = re.sub(r'\bi\b', 'I', s)
    return s[0].upper()+s[1:]

import json
json.dumps([(meta, [fixup(s) for s in sents]) for meta, sents in manual_bos.groups])
