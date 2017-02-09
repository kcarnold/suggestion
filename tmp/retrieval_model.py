# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:35:29 2017

@author: kcarnold
"""

from suggestion import suffix_array
import joblib
import ujson as json
#%%
docs = json.load(open('models/tokenized_reviews.json'))
#%%
sufarr = suffix_array.DocSuffixArray(docs=docs, **joblib.load('models/yelp_sufarr.joblib'))
#sufarr = suffix_array.DocSuffixArray(docs=docs, **joblib.load('../megacomplete/yelp_sufarr.joblib'))
#%%
from suggestion import suggestion_generator
model = suggestion_generator.get_model('yelp_train')
#%%

context = 'my friends '
context_toks = suggestion_generator.tokenize_sofar(context)
#%%
a, b = sufarr.search_range((context_toks[-1], ''))
def collect_words_in_range_slow(start, after_end, word_idx):
    words = set()
    for idx in range(start, after_end):
        words.add(sufarr.get_suffix_by_idx(idx)[word_idx])
    return words
set_a = collect_words_in_range_slow(a, b, 1)
#%%
def collect_words_in_range(start, after_end, word_idx):
    words = set()
    words.add(sufarr.get_suffix_by_idx(start)[word_idx])
    for i in range(start, after_end):
        if sufarr.lcp[i] <= word_idx:
            word = sufarr.get_suffix_by_idx(i + 1)[word_idx]
            print(i, word)
            assert word not in words
            words.add(word)
    return words
set_b = collect_words_in_range(a, b, 1)

end = a
assert collect_words_in_range_slow(a, end, 1) == collect_words_in_range(a, end, 1)
#%%

for idx in [7330997, 7332731,7332764,7332765]:
    print(idx - a, sufarr.get_suffix_by_idx(idx+1)[:5])
#%%
for i in range(a, b):
    assert tuple(sufarr.get_suffix_by_idx(i)[:10]) <= tuple(sufarr.get_suffix_by_idx(i+1)[:10]), i

#%%
vocab_indices =
model.
#%%
