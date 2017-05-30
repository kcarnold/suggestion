# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:48:33 2017

@author: kcarnold
"""
import numpy as np
from suggestion import suggestion_generator
import kenlm
import tqdm
#%%

def sizeof_fmt(num, suffix='B', units=None, power=None, sep='', precision=2, sign=False):
    prefix = '+' if sign and num > 0 else ''

    for unit in units[:-1]:
        if abs(round(num, precision)) < power:
            if isinstance(num, int):
                return "{}{}{}{}{}".format(prefix, num, sep, unit, suffix)
            else:
                return "{}{:3.{}f}{}{}{}".format(prefix, num, precision, sep, unit, suffix)
        num /= float(power)
    return "{}{:.{}f}{}{}{}".format(prefix, num, precision, sep, units[-1], suffix)


def sizeof_fmt_iec(num, suffix='B', sep='', precision=2, sign=False):
    return sizeof_fmt(num, suffix=suffix, sep=sep, precision=precision, sign=sign,
                      units=['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi'], power=1024)


def sizeof_fmt_decimal(num, suffix='B', sep='', precision=2, sign=False):
    return sizeof_fmt(num, suffix=suffix, sep=sep, precision=precision, sign=sign,
                      units=['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'], power=1000)
#%%
model = suggestion_generator.get_model('yelp_train')
start_state = model.get_state([','])[0]
sufarr = suggestion_generator.sufarr
a, b = sufarr.search_range((',', ''))
sizeof_fmt_decimal(b-a)
#%%
%timeit suggestion_generator.collect_words_in_range(a,b,1)
#%%
unigram_probs = model.unigram_probs
lookup_voc = model.model.vocab_index
rarest_words_flat = []
for doc in tqdm.tqdm(sufarr.docs):
    word_indices = [lookup_voc(w) for w in doc]
    ups = unigram_probs[word_indices]
    rarest_words_flat.extend([np.nanmin(ups[i+1:i+6]) for i in range(len(doc)-1)])
    rarest_words_flat.append(0.) # for the last token in the document
    rarest_words_flat.append(0.) # for the end-of-document token
rarest_words_flat = np.array(rarest_words_flat)
#%%
words_before_eos = []
for doc in tqdm.tqdm(sufarr.docs):
    eos_indices = [idx for idx, tok in enumerate(doc) if tok == '</S>'] + [len(doc)]
    offset = 0
    cur_eos_idx = eos_indices[0]
    for i, tok in enumerate(doc):
        until_eos = cur_eos_idx - i
        if until_eos < 0:
            offset += 1
            cur_eos_idx = eos_indices[offset]
            until_eos = cur_eos_idx - i
        words_before_eos.append(until_eos)
    words_before_eos.append(0) # for end of document token
words_before_eos = np.array(words_before_eos)
#%%
# A filtered suffix array is one where only a subset of the possible indices exist in the lookup tables.
# To construct one, we create a mask over the existing indices.
# The suffix_array array maps from sorted index to master location.
# doc_idx is the document idx for the
filtered_doc_idx = sufarr.doc_idx[(words_before_eos > 5)[sufarr.suffix_array[1:]]]
filtered_tok_idx = sufarr.tok_idx[(words_before_eos > 5)[sufarr.suffix_array[1:]]]
#%%
idx = 1000
sufarr.docs[filtered_doc_idx[idx]][filtered_tok_idx[idx]:][:10]
#%%
from suggestion.suffix_array import DocSuffixArray
filtered_sufarr = DocSuffixArray(sufarr.docs, None, filtered_doc_idx, filtered_tok_idx, None)
#%%
a, b = filtered_sufarr.search_range(('the', ''))
import random
filtered_sufarr.get_partial_suffix(random.randrange(a,b), 0, 10)
#%%
rare_word_raw = np.array([tok for rw in rarest_words_by_doc for tok in rw + [0]])
#%%
docs_flat_raw = []
for doc in tqdm.tqdm(sufarr.docs):
    docs_flat_raw.extend(doc)
    docs_flat_raw.append('</d>')
#docs_flat_raw = [tok for doc in sufarr.docs for tok in doc + ['</d>']])
#%%
%timeit rarest_words_by_sufarr_idx = rare_word_raw[sufarr.suffix_array[a:b] + 1]
#[rarest_words_by_doc[sufarr.doc_idx[idx]][sufarr.tok_idx[idx] + 1] for idx in range(a,b)]
#%%

%timeit for idx in range(a,b): offset = sufarr.suffix_array[idx]; phrase = docs_flat_raw[offset:offset+5]
#%%
%timeit for idx in range(a,b): sufarr.get_partial_suffix(idx, 1, 5)
#%%
context_words = 1
N_EVAL = 3
while True:
    phrase = sufarr.get_partial_suffix(a, context_words, context_words + N_EVAL)
    if len(phrase) < N_EVAL:
        a += 1
    else:
        break
states = [start_state]
scores = [0.]
while len(states) < N_EVAL + 1:
    state = kenlm.State()
    score = model.model.BaseScore(states[-1], phrase[len(states) - 1], state)
    scores.append(scores[-1] + score)
    states.append(state)
#%%
skipped = 0
lcp = suggestion_generator.sufarr.lcp
for idx in tqdm.tqdm(range(a+1, b)):
    in_common = lcp[idx-1] - context_words
    new_phrase = sufarr.get_partial_suffix(idx, context_words, context_words + N_EVAL)
    states[in_common+1:] = []
    scores[in_common+1:] = []
    if len(new_phrase) < N_EVAL or '</S>' in new_phrase:
        skipped += 1
        continue
    while len(states) < N_EVAL + 1:
        state = kenlm.State()
        score = 0#model.model.BaseScore(states[-1], phrase[len(states) - 1], state)
        scores.append(scores[-1] + score)
        states.append(state)
#    assert scores[-1] * suggestion_generator.LOG10 == model.score_seq(start_state, phrase[:N_EVAL])[0]
#%%
for idx in tqdm.tqdm(range(a+1, b)):
    model.score_seq(start_state, sufarr.get_partial_suffix(idx, context_words, context_words + N_EVAL))[0]