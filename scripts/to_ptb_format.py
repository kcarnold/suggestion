# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:37:21 2017

@author: kcarnold
"""

import pickle
import tqdm
#%%
vocab, counts = pickle.load(open('yelp_preproc/all_data.pkl','rb'))['vocab']
#%%
tok2id = {tok: idx for idx, tok in enumerate(vocab[:25000])}
#%%
def unkify(text):
    return ' '.join(tok if tok in tok2id else '<unk>' for tok in text.split())
unkify('the aoeustnh ihe gone')
#%%
def proc_fold(name):
    fold = pickle.load(open(f'yelp_preproc/{name}_data.pkl','rb'))['data']
    with open(f'yelp_preproc/{name}.txt', 'w') as f:
        for tokenized in tqdm.tqdm(fold.tokenized, desc=name):
            for sent in tokenized.lower().split('\n'):
                if sent.strip():
                    f.write(' ' + unkify(sent) + '\n')

proc_fold('train')
#%%
proc_fold('valid')
proc_fold('test')
