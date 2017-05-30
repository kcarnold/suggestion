# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:55:28 2017

@author: kcarnold
"""
import numpy as np
from suggestion import suggestion_generator
#%%
sofar = ''
model = suggestion_generator.get_model('yelp_train')
toks = suggestion_generator.tokenize_sofar(sofar)

#%%
word_bonuses = np.zeros(len(model.id2str))
for word in prewrite.split():#['burrito', 'chicken', 'salsa']:
    idx = model.model.vocab_index(word)
    if idx != 0:
        word_bonuses[idx] = 50.
#%%
sofar = ''#'chicken burrito '
for i in range(50):
    phrases = suggestion_generator.get_suggestions(sofar=sofar, cur_word=[], domain='yelp_train', rare_word_bonus=1., temperature=0., use_bos_suggs=False, use_sufarr=True, word_bonuses=word_bonuses)[0]
    if len(phrases) == 0:
        word = '.'
    else:
        word = phrases[0][0][0]
    sofar = sofar + word + ' '
    print(sofar)
#%%
set(prewrite.split()) - set(sofar.split())



#%%
print('; '.join(' '.join(s) for s, p in
                suggestion_generator.get_suggestions(sofar='when  ', cur_word=[], domain='yelp_train', rare_word_bonus=1., temperature=0., use_bos_suggs=False, use_sufarr=True, word_bonuses=None, length_after_first=10, null_logprob_weight=0)[0]
                ))