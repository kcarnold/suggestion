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
clf = suggestion_generator.sentiment_classifier
#%%
lang_models = [suggestion_generator.get_model(f'yelp_train-{star}star') for star in [5, 3, 1]]
#%%
def get_sentiment_diverse_bos(sofar, toks, sug_state, *, domain='yelp_train', length_after_first=17):
    """
    Get beginning-of-sentence suggestions that are diverse in sentiment and not too repetitive.

    Approach: generate from 5-star, 3-star, and 1-star LMs, but ensure diversity of first word
    with respect to (1) the other slots and (2) the prior words used by the same LM.
    """
    beam_search_kwargs = {'constraints': {}}

    already_used_first_words = sug_state.get('already_used_first_words')
    if already_used_first_words is None:
        sug_state['already_used_first_words'] = already_used_first_words = [set() for _ in lang_models]
    first_word_ents = [suggestion_generator.beam_search_phrases(
                    lang_model, toks, beam_width=50, length_after_first=1, **beam_search_kwargs)
        for lang_model in lang_models]
    valid_first_words = [
            (ent.score, slot, ent.words[0], ent)
            for slot, (already_used, ents) in enumerate(zip(already_used_first_words, first_word_ents))
            for ent in ents
            if ent.words[0] not in already_used]
    valid_first_words.sort(reverse=True)
    # FIXME: what if there are none?
    first_words = [None] * 3
    first_ents = [None] * 3
    slots_assigned = 0

    # Assign first words in descending order of overall likelihood.
    for score, slot, word, ent in valid_first_words:
        if first_words[slot] is not None:
            # Already got this slot.
            continue
        if word in first_words:
            # Already claimed, sorry.
            continue
        # Yay, found an unused okay word!
        first_words[slot] = word
        first_ents[slot] = ent
        slots_assigned += 1
        already_used_first_words[slot].add(word)
        if slots_assigned == len(first_words):
            break
    if slots_assigned != len(first_words):
        # Um... oops.
        print("Oops, no valid first-words to assign :(")
        return [], sug_state

    # Now just beam-search forward from each.
    beam_search_results = [suggestion_generator. beam_search_phrases_loop(model, [ent],
            start_idx=1,
            beam_width=50,
            length_after_first=length_after_first, **beam_search_kwargs)
        for model, ent in zip(lang_models, first_ents)]
    phrases = [(ent.words, dict(score=ent.score, type='bos')) for ents in beam_search_results for ent in ents[:1]]
    return phrases, sug_state

sug_state = {}
sofar = ''
for i in range(10):
    toks = suggestion_generator.tokenize_sofar(sofar)
    clf_state = clf.get_state(toks)
    phrases, sug_state = get_sentiment_diverse_bos(sofar, toks, sug_state)
    print('{:30s} {:30s} {:30s}'.format(*[' '.join(phr) for phr, meta in phrases]))
    print('{:<30.2f} {:<30.2f} {:<30.2f}'.format(*[clf.sentiment(clf_state, phr) for phr, meta in phrases]))
    print()
    sofar += ' i used to come here every week. '
#%%
toks = suggestion_generator.tokenize_sofar(sofar)
phrases, sug_state = get_sentiment_diverse_bos(sofar, toks, sug_state)