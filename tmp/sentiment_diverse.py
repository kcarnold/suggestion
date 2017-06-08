# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:54:58 2017

@author: kcarnold
"""
#%%
import numpy as np
#%%
from suggestion import suggestion_generator
from scipy.special import expit
clf = suggestion_generator.CLASSIFIERS['positive']
#%%
from suggestion.lang_model import LMClassifier
clf = LMClassifier([
    suggestion_generator.get_model(f'yelp_train-{star}star') for star in [1, 2, 4, 5]], [-.5, -.5, .5, .5])
clf.classify_seq(clf.get_state([]), "i wouldn't recommend this place".split())
#%%
domain = 'yelp_train'
model = suggestion_generator.get_model(domain)
prefix_logprobs = None
constraints = {}
sentence_enders = []
length_after_first = 17


sofar = 'i know '
toks = suggestion_generator.tokenize_sofar(sofar)
clf_startstate = clf.get_state(toks)

from suggestion import diversity
phrases = diversity.get_suggs_with_sentiment_diversity(model, toks, clf=clf)
print(phrases)