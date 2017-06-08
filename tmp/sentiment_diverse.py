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
#import itertools
#def diversity(scalars):
#    return np.sort([(np.abs(a - b)) for a, b in itertools.combinations(scalars, 2)])[-2]
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
def diversity(x, max_distance=1.):
    x = np.array(x)[:,None]
    K = max_distance - squareform(pdist(x))
#    K = cosine_similarity(x)
    return np.linalg.det(K)

diversity([.3, .5, .9]), diversity([.01, .99, .999]), diversity([.01, .99, .5])
#%%
domain = 'yelp_train'
prefix_logprobs = None
constraints = {}
sentence_enders = []
length_after_first = 17


sofar = 'i know '
toks = suggestion_generator.tokenize_sofar(sofar)
clf_startstate = clf.get_state(toks)

# Branch 5 ways for the first word
N_initial = 10
first_word_ents = suggestion_generator.beam_search_phrases(domain, toks, beam_width=N_initial, length_after_first=1, prefix_logprobs=prefix_logprobs, constraints=constraints)

classified_ents = [(ent[0], clf.classify_seq(clf_startstate, ent[1]), ent[1]) for ent in first_word_ents]
classified_ents.sort(reverse=True)
for llk, pos, words in classified_ents:
    print(f"{pos:3.2f} {llk:6.2f} {' '.join(words)}")
print()
print()
# For each first-word branch, pick 10.
N_post = 5
model = suggestion_generator.get_model('yelp_train')
ents = []
for fwent in first_word_ents:
    beam = [fwent]
    for iteration_num in range(1, length_after_first):
        beam = suggestion_generator.beam_search_phrases_extend(model, beam, iteration_num=iteration_num, beam_width=N_post, length_after_first=length_after_first,
            prefix_logprobs=prefix_logprobs, rare_word_bonus=0., constraints=constraints)
        prefix_logprobs = None
    ents.extend(beam)

# Pick a set of phrases that
# - has 3 different first words
# - each is reasonably likely
# - there's a diversity of sentiments, if possible.
# -> approach: do the first two, then patch for the third.

classified_ents = [(ent[0], clf.classify_seq(clf_startstate, ent[1]), ent[1]) for ent in ents]
#negs = [i for i, ent in classified_ents if ]
classified_ents.sort(reverse=True)
suggs = []
first_words_used = {}
for llk, pos, words in classified_ents:
    first_word = words[0]
    if first_word in first_words_used:
        continue
    first_words_used[first_word] = len(suggs)
    suggs.append((llk, pos, words))
#    print(f"{pos:3.2f} {llk:6.2f} {' '.join(words)}")
    if len(suggs) == 3:
        break

cur_sentiments = np.array([pos for llk, pos, words in suggs])
cur_sentiment_diversity = diversity(cur_sentiments)
max_logprob_penalty = -1.
min_logprob_allowed = min(llk for llk, pos, words in suggs) + max_logprob_penalty

# Greedily replace suggestions so as to increase sentiment diversity.
while True:
    for llk, pos, words in suggs:
        print(f"{pos:3.2f} {llk:6.2f} {' '.join(words)}")
    print()
    print()

    cur_sentiment_mean = np.mean(cur_sentiments)
    contribution_to_diversity = (cur_sentiments - cur_sentiment_mean) ** 2
    least_diverse = np.argmin(contribution_to_diversity)

    candidates = []
    for llk, pos, words in classified_ents:
        if llk < min_logprob_allowed:
            continue
        # Would this increase the sentiment diversity if we added it?
        # Case 1: it replaces an existing word
        replaces_slot = first_words_used.get(words[0])
        if replaces_slot is not None:
            candidate_sentiments = cur_sentiments.copy()
            candidate_sentiments[replaces_slot] = pos
            new_diversity = diversity(candidate_sentiments)
        else:
            # Case 2: it replaces the currently least-diverse word.
            new_diversities = np.empty(3)
            for replaces_slot in range(3):
                candidate_sentiments = cur_sentiments.copy()
                candidate_sentiments[replaces_slot] = pos
                new_diversities[replaces_slot] = diversity(candidate_sentiments)
            # Check this, I'm lazy.
            replaces_slot = np.argmax(new_diversities)
#            assert replaces_slot == least_diverse
            new_diversity = new_diversities[replaces_slot]
        if new_diversity > cur_sentiment_diversity:
            candidates.append((new_diversity, replaces_slot, llk, pos, words))
    print(f"Found {len(candidates)} candidates that increase diversity")
    if len(candidates) == 0:
        break
    prev_sentiment_diversity = cur_sentiment_diversity
    cur_sentiment_diversity, replaces_slot, llk, pos, words = max(candidates)
    existing_sugg = suggs[replaces_slot]
    print(f"Replacing slot {replaces_slot} with llk={llk:.2f} pos={pos:.2f} \"{' '.join(words)}\" to gain {cur_sentiment_diversity - prev_sentiment_diversity:.2f} diversity")
    # Actually replace the suggestion.
    cur_sentiments[replaces_slot] = pos
    assert cur_sentiment_diversity == diversity(cur_sentiments)
    del first_words_used[existing_sugg[2][0]]
    first_words_used[words[0]] = replaces_slot
    suggs[replaces_slot] = (llk, pos, words)




#%%


next_words = sentence_enders + [ent.words[0] for ent in first_word_ents[:N_initial]]
while len(next_words) < N_initial:
    next_words.append(None)
phrases = [([], None)] * N_initial
slots = []
jobs = []
for slot, next_word in enumerate(next_words[:N_initial]):
    slots.append(slot)

    jobs.append(suggestion_generator.predict_forward(
        domain, toks + [next_word], beam_width=50,
        length_after_first=length_after_first, constraints=constraints))
results = jobs
for slot, result in zip(slots, results):
    phrases[slot] = result

clf_scores = [clf.classify_seq(clf_startstate, phrase) for phrase, _ in phrases]
list(zip(clf_scores, phrases))