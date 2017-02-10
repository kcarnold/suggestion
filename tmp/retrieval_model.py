# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:35:29 2017

@author: kcarnold
"""

import numpy as np
from suggestion import suffix_array
import joblib
import ujson as json
import kenlm
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
    if start == after_end:
        return words
    words.add(sufarr.get_suffix_by_idx(start)[word_idx])
    for i in range(start + 1, after_end):
        # Invariant: words contains all words at offset word_idx in suffixes from
        # start to i.
        if sufarr.lcp[i - 1] <= word_idx:
            word = sufarr.get_suffix_by_idx(i)[word_idx]
#            print(i, word)
            assert word not in words
            words.add(word)
    return words
#set_b = collect_words_in_range(a, b, 1)

end = b
assert collect_words_in_range_slow(a, end, 1) == collect_words_in_range(a, end, 1)
#%%
def generate_phrase_from_sufarr(model, sufarr, context_toks, length, temperature=1.):
    if context_toks[0] == '<s>':
        state, _ = model.get_state(context_toks[1:], bos=True)
    else:
        state, _ = model.get_state(context_toks, bos=False)
    start_idx, end_idx = sufarr.search_range((context_toks[-1], ''))
    phrase = []
    generated_logprobs = np.empty(length)
    for i in range(length):
        next_words = sorted(collect_words_in_range(start_idx, end_idx, i + 1))
        if len(next_words) == 0:
            raise suggestion_generator.GenerationFailedException
        vocab_indices = [model.model.vocab_index(word) for word in next_words]
        logprobs = model.eval_logprobs_for_words(state, vocab_indices)
        logprobs /= temperature
        probs = suggestion_generator.softmax(logprobs)

        picked_subidx = np.random.choice(len(probs), p=probs)
        picked_idx = vocab_indices[picked_subidx]
        new_state = kenlm.State()
        model.model.base_score_from_idx(state, picked_idx, new_state)
        state = new_state
        word = next_words[picked_subidx]
#        assert word == model.id2str[picked_idx]
        phrase.append(word)
        generated_logprobs[i] = np.log(probs[picked_subidx])
        start_idx, end_idx = sufarr.search_range((context_toks[-1],) + tuple(phrase) + ('',))
        print(start_idx, end_idx-start_idx)
    return phrase, generated_logprobs

generate_phrase_from_sufarr(model, sufarr, context_toks, 30)
#%%
