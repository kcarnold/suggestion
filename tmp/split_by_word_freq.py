# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:20:57 2017

@author: kcarnold
"""

from suggestion import analyzers
from suggestion import suggestion_generator
from suggestion.paths import paths
import pickle
import numpy as np
from scipy.spatial.distance import pdist
import tqdm
import pandas as pd

#%%
reviews = analyzers.load_reviews()
#%%
wordfreq_analyzer = analyzers.WordFreqAnalyzer.build()
#%%
#wordpair_analyzer = pickle.load(open(paths.models / 'wordpair_analyzer.pkl', 'rb'))
#%%
sentences = [(doc_idx, sent_idx, sent) for doc_idx, doc in enumerate(tqdm.tqdm(reviews.tokenized)) for sent_idx, sent in enumerate(doc.split('\n'))]
#%%
sent_doc_idx, sent_sent_idx, sent_text = zip(*sentences)
sent_doc_idx = np.array(sent_doc_idx)
sent_sent_idx = np.array(sent_sent_idx)
sent_text_raw = list(sent_text)
#%%
from suggestion import train_ngram
sent_text = [train_ngram.cant_type.sub('', train_ngram.sub_numbers(text.lower())) for text in tqdm.tqdm(sent_text_raw, desc="Convert to typeable form")]

#%%
word_freqs = []
for i, toks in enumerate(tqdm.tqdm(sent_text, desc="Compute mean word llks")):
    toks = toks.lower().split()
    toks = [tok for tok in toks if tok[0] not in '.,?!:-<']
    indices = wordfreq_analyzer.lookup_indices(toks)
    if len(indices):
        unifreqs = wordfreq_analyzer.log_freqs[indices]
        wf_mean = np.mean(unifreqs)
        wf_std = np.std(unifreqs)
    else:
        wf_mean = wf_std = np.nan
    word_freqs.append(wf_mean)
word_freqs = np.array(word_freqs)
#%%
contextual_llks = []
model = suggestion_generator.get_model('yelp_train-balanced')
start_states = [model.get_state(['<s>', '<D>'])[0], model.get_state(["<s>", "<S>"])[0]]
for i, toks in enumerate(tqdm.tqdm(sent_text, desc="Compute mean contextual llks")):
    toks = toks.lower().split()
    start_state = start_states[min(sent_sent_idx[i], 1)]
    scores = model.score_seq_by_word(start_state, toks)
    contextual_llks.append(np.mean(scores))
contextual_llks = np.array(contextual_llks)
#%%
valid_wordfreq = ~np.isnan(word_freqs)
token_lengths = np.array([len(sent.split()) for sent in sent_text])
min_length, max_length = np.percentile(token_lengths, [10, 90])
long_enough = (token_lengths >= min_length) & (token_lengths <= max_length)
indices_2 = np.flatnonzero(valid_wordfreq & long_enough)
sent_doc_idx_2 = sent_doc_idx[indices_2]
sent_sent_idx_2 = sent_sent_idx[indices_2]
sent_text_2 = [sent_text[i] for i in indices_2]
word_freqs_2 = word_freqs[indices_2]
contextual_llks_2 = contextual_llks[indices_2]
#%%
[(i, sent_text_2[i], word_freqs_2[i]) for i in np.argsort(word_freqs_2)[-50:]]
#%%
[(i, sent_text_2[i], word_freqs_2[i]) for i in np.argsort(contextual_llks_2)[-50:]]
#%%
[(i, sent_text_2[i], word_freqs_2[i]) for i in np.argsort(contextual_llks_2)[:50]]
#%%
import seaborn as sns
sns.distplot(word_freqs_2)
#%%
sns.distplot(contextual_llks_2)
#%%

split_point = np.median(word_freqs_2)
rs = np.random.RandomState(0)
[(sent_text_2[i]) for i in rs.choice(np.flatnonzero(word_freqs_2 > split_point), 10, replace=False)]

#%%
# Train LMs
from suggestion.util import dump_kenlm
def dump_indices(name, indices):
    dump_kenlm(name, sorted({("<D> " if sent_sent_idx_2[i] == 0 else "<S> ") + sent_text_2[i].lower() + " </S>" for i in tqdm.tqdm(indices, desc=name)}))
#%%
dump_indices('yelp_lowfreq', np.flatnonzero(word_freqs_2 < split_point))
dump_indices('yelp_hifreq', np.flatnonzero(word_freqs_2 > split_point))
dump_indices('yelp_allsentfilt', np.arange(len(sent_text_2)))
#%%
split_point_contextual = np.median(contextual_llks_2)
rs = np.random.RandomState(0)
dump_indices('yelp_lowllk', np.flatnonzero(contextual_llks_2< split_point_contextual))
dump_indices('yelp_hillk', np.flatnonzero(contextual_llks_2 > split_point_contextual))


#%%
from suggestion.lang_model import Model
lowfreq_model = Model.get_or_load_model('yelp_lowfreq')
hifreq_model = Model.get_or_load_model('yelp_hifreq')
all_model = Model.get_or_load_model('yelp_allsentfilt')
#%%
models = [lowfreq_model, hifreq_model, all_model]
scores = np.array([[model.score_seq(model.bos_state, k.lower().split())[0] for model in models] for k in tqdm.tqdm(sent_text_2)])
#%%
cat_scores, all_scores = scores[:,:2], scores[:,2]
from scipy.special import logsumexp
discrim = cat_scores - logsumexp(cat_scores, axis=1, keepdims=True)
[sent_text_2[idx] for idx in np.argsort(discrim[:,0])[-10:]]

#%%
suggestion_generator.get_suggestions(sofar="the food is ", cur_word=[], domain='yelp_hifreq', rare_word_bonus=0., use_sufarr=False, temperature=0., use_bos_suggs=False)