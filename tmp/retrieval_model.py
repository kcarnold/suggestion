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
#%%
from suggestion import suggestion_generator
model = suggestion_generator.get_model('yelp_train')
#%%
def collect_words_in_range_slow(start, after_end, word_idx):
    words = set()
    for idx in range(start, after_end):
        words.add(sufarr.get_suffix_by_idx(idx)[word_idx])
    return words

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
#assert collect_words_in_range_slow(a, end, 1) == collect_words_in_range(a, end, 1)
#%%
import scipy.stats
def generate_phrase_from_sufarr(model, sufarr, context_toks, length, temperature=1.):
    if context_toks[0] == '<s>':
        state, _ = model.get_state(context_toks[1:], bos=True)
    else:
        state, _ = model.get_state(context_toks, bos=False)
    phrase = []
    generated_logprobs = np.empty(length)
    for i in range(length):
        start_idx, end_idx = sufarr.search_range((context_toks[-1],) + tuple(phrase) + ('',))
        next_words = sorted(collect_words_in_range(start_idx, end_idx, i + 1))
        if len(next_words) == 0:
            raise suggestion_generator.GenerationFailedException
#        if len(next_words) == 1:
            # We
        vocab_indices = [model.model.vocab_index(word) for word in next_words]
        logprobs = model.eval_logprobs_for_words(state, vocab_indices)
        logprobs /= temperature
        probs = suggestion_generator.softmax(logprobs)
#        if len(next_words) < 10:
#            print(next_words, probs)
#        print(start_idx, end_idx-start_idx, len(next_words), scipy.stats.entropy(probs))

        picked_subidx = np.random.choice(len(probs), p=probs)
        picked_idx = vocab_indices[picked_subidx]
        new_state = kenlm.State()
        model.model.base_score_from_idx(state, picked_idx, new_state)
        state = new_state
        word = next_words[picked_subidx]
#        assert word == model.id2str[picked_idx]
        phrase.append(word)
        generated_logprobs[i] = np.log(probs[picked_subidx])
    return phrase, generated_logprobs


for context in ['', 'my', 'the lunch menu', 'we', 'i could not imagine', 'absolutely', "i love", "my first",]:
    context_toks = suggestion_generator.tokenize_sofar(context + ' ')
    print('\n\n', context)
    print("- Exploratory")
    for i in range(5):
        print('', ' '.join(generate_phrase_from_sufarr(model, sufarr, context_toks, 6, temperature=1)[0]))
    print("- Max likelihood")
    print('\n'.join([' ' + ' '.join(x['words'])
        for x in suggestion_generator.beam_search_phrases(model, context_toks, 100, 30)[:5]]))
#%%

# Simplest thing to do: go back until we have < K phrases, then pick one at random.
def draw_randomly_from_context_match(sufarr, context_toks, min_suffixes):
    a, b = 0, len(sufarr.tok_idx)
    context_toks_idx = len(context_toks)
    while context_toks_idx > 0:
        search_phrase = tuple(context_toks[context_toks_idx - 1:])
        new_a, new_b = sufarr.search_range(search_phrase)
        print(search_phrase, new_a, new_b)
        if new_b - new_a > min_suffixes:
            a = new_a
            b = new_b
            context_toks_idx -= 1
        else:
            print(a, b, search_phrase)
            break
    return sufarr.get_suffix_by_idx(np.random.choice(range(a, b)))
' '.join(draw_randomly_from_context_match(sufarr, suggestion_generator.tokenize_sofar('i love '), 100)[:10])
#%%
def search_context(sufarr, context, try_to_get_less_than, min_suffixes=2):
    offset_into_suffix = 1
    while True:#offset_into_suffix <= len(context):
        search_for = context[-offset_into_suffix:]
        a, b = sufarr.search_range(tuple(search_for))
        num_found = b - a
#        print(num_found, search_for)
        if num_found < min_suffixes:
#            print("Too few")
            offset_into_suffix -= 1
            a, b = sufarr.search_range(tuple(search_for)[1:])
            break
        if b - a < try_to_get_less_than:
#            print("Ok", offset_into_suffix)
            break
        if offset_into_suffix < len(context):
            offset_into_suffix += 1
        else:
            break
    return a, b, offset_into_suffix
a, b, offset = search_context(sufarr, '<D> great food ,'.split() + [''], 100)
offset -= 1
print(sufarr.get_suffix_by_idx(a)[:offset])
print(sufarr.get_suffix_by_idx(a)[offset:][:5])

#%%
incorrect = []
num_alternatives_es = []
for i in range(1000):
    while True:
        doc_idx = np.random.choice(len(docs))
        doc_toks = docs[doc_idx]
        tok_idx = np.random.randint(1, len(doc_toks))
        context = doc_toks[:tok_idx]
        true_suffix = doc_toks[tok_idx:]
        if context[-1][0] == '<' or true_suffix[0][0] == '<':
            # TODO: also a problem if true_suffix doesn't include any real words. Or if it's way too common, like ['.', '</S>'].
            continue
        break


    context_state = model.get_state(context[-6:], bos=False)[0]

    a, b, offset = search_context(sufarr, context + [''], try_to_get_less_than=100)
    offset -= 1
    if offset == 0:
        print("Oops, didn't find context??")
        continue
    assert sufarr.get_suffix_by_idx(a)[:offset] == context[-offset:]
    truesuf_a, truesuf_b = sufarr.search_range(tuple(context[len(context)-offset:] + true_suffix))
    assert truesuf_b - truesuf_a >= 1
    assert sufarr.get_suffix_by_idx(truesuf_a)[:offset] == context[-offset:]
    if a == truesuf_a and b == truesuf_b:
        print("Oops, unique word??")
        continue

    num_alternatives_es.append(b - a - (truesuf_b - truesuf_a))
    while True:
        c = np.random.randint(a, b)
        if truesuf_a <= c < truesuf_b:
            continue
        break
    # TODO: instead, compute the rank of the correct suffix compared with the alternatives.
    # If there are a ton of alternatives, take a random sample of a fixed-size subset of them.
    c_suf = sufarr.get_suffix_by_idx(c)[offset:]
    #print(sufarr.docs[sufarr.doc_idx[c]][sufarr.tok_idx[c]-offset:][:10])
    assert sufarr.get_suffix_by_idx(c)[:offset] == context[-offset:]
    true_score = model.score_seq(context_state, true_suffix[:5])[0]
    fake_score = model.score_seq(context_state, c_suf[:5])[0]
    # TODO: another thing to try: look at the difference between the likelihood in the originial context and the likelihood in the new context.
    if fake_score > true_score:
        incorrect.append((doc_idx, tok_idx, c))
#    print(true_score, ' '.join(context[-5:]), '|', ' '.join(true_suffix[:5]))
#    print(fake_score, ' '.join(context[-5:]), '|', ' '.join(c_suf[:5]))
