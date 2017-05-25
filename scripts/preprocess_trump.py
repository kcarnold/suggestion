# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:16:58 2017

@author: kcarnold
"""

import nltk
import pathlib
import pandas as pd
from nltk.tokenize.casual import _replace_html_entities, remove_handles
from suggestion.tokenization import URL_RE, token_spans
#%%
tweetcorpus_path = pathlib.Path('~/Data/trump_tweet_data_archive')
tweets = pd.concat((pd.read_json(tweetcorpus_path / 'condensed_2017.json'), pd.read_json(tweetcorpus_path/ 'condensed_2016.json')), ignore_index=True)
not_rts = tweets.text[~tweets.is_retweet]

#%%
def preprocess(tweet_text):
    return URL_RE.sub(' ', remove_handles(_replace_html_entities(tweet_text)))

from suggestion import train_ngram
def tokenize(text):
#    text = URL_RE.sub(' ', text)
    sents = nltk.sent_tokenize(text)
    # Use our simple word tokenizer, since spacy breaks apart contractions.
    token_spaced_sents = (' '.join(sent[a:b] for a, b in token_spans(sent)) for sent in sents)
    return '\n'.join(token_spaced_sents)


#%%
import tqdm
from suggestion.util import dump_kenlm
dump_kenlm('tweeterinchief', (' '.join(train_ngram.convert_tokenization(tokenize(preprocess(text)))) for text in tqdm.tqdm(not_rts)))
