# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:12:34 2017

@author: kcarnold
"""

from suggestion.analyzers import load_reviews
#%%
reviews = load_reviews()
#%%
import re
def strip(x):
    return re.sub(r'[^a-z]', '', x.lower())

businesses = reviews.drop_duplicates('business_id').set_index('business_id').name
#%%
chipotles = [biz_id for biz_id, name in businesses.items() if 'chipotle' in name.lower()]
chipotle_reviews = reviews[reviews.business_id.isin(chipotles)]
import numpy as np

texts = np.random.RandomState(0).choice(chipotle_reviews.text, 10, replace=False).tolist()

import json
json.dumps(texts)

#%%
[(biz_id, name) for biz_id, name in businesses.items() if 'panera' in name.lower()]
[(biz_id, name) for biz_id, name in businesses.items() if 'outbacksteak' in strip(name)]
[(biz_id, name) for biz_id, name in businesses.items() if 'texasroad' in strip(name)]
#%%
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1,3), min_df=10)
vecs = vectorizer.fit_transform(businesses.values)
sorted(zip(vecs.sum(axis=0).A1, vectorizer.get_feature_names()))
#%%
# For each term, how geographically diverse is it?
bizdata = reviews.drop_duplicates('business_id').set_index('business_id')
geogs = bizdata.city.str.cat(bizdata.state, sep=', ')
geog_vectorizer = CountVectorizer(analyzer=lambda x:x)
geog_vecs = geog_vectorizer.fit_transform(geogs.values)
term_by_geog = vecs.T.dot(geog_vecs)
term_by_geog.data.fill(1)
sorted(zip(term_by_geog.sum(axis=1).A1, vectorizer.get_feature_names()))