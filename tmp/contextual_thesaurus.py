# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:39:01 2017

@author: kcarnold
"""

import numpy as np
#%%
from suggestion import suggestion_generator, clustering
#%%
#from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise
#%%
model = suggestion_generator.get_model('yelp_train-balanced')
model_zipfs = model.unigram_probs / np.log(10) + 9
#%%
cnnb = clustering.ConceptNetNumberBatch.load()
#%%
def get_vecs_for_words(cnnb, words):
    res = np.zeros((len(words), cnnb.ndim))
    for i, word in enumerate(words):
        try:
            res[i] = cnnb[word]
        except KeyError:
            pass
    return res
all_vecs = get_vecs_for_words(cnnb, model.id2str)
#%%
#word_clusters = KMeans(n
#%%
for sofar in ['the service was ', 'the food was ', 'i loved the ', 'i ', 'the ', "we could "]:
    print(sofar)
    toks = suggestion_generator.tokenize_sofar(sofar)
    state = model.get_state(toks)[0]
    next_words, logprobs = model.next_word_logprobs_raw(state, toks[-1])
    vecs_for_words = all_vecs[next_words]
    top_3 = np.argsort(logprobs)[-3:][::-1]
    sim_to_top_3 = pairwise.cosine_similarity(vecs_for_words[top_3], vecs_for_words)
    # for future ref (https://www.researchgate.net/publication/282359004_HANDLING_THE_IMPACT_OF_LOW_FREQUENCY_EVENTS_ON_CO-OCCURRENCE_BASED_MEASURES_OF_WORD_SIMILARITY_KDIR_2011_KDIR-_International_Conference_on_Knowledge_Discovery_and_Information_Retrieval_226-231_Paris_O)
    # exp(PMI2) = p(x,y)**2 / (p(x)p(y)) = p(y|x)/p(y) * p(x,y) = p(y|x)/p(y) * p(y|x)p(x)
    relevance = logprobs - .5 * model.unigram_probs[next_words]
    for sims in sim_to_top_3:
        candidates = np.argsort(sims)[-10:][::-1]
        relevances = relevance[candidates]
        print(model.id2str[next_words[candidates[0]]], '-', ', '.join('{}[{:.2f}]'.format(model.id2str[next_words[idx]], relevance[idx]) for idx in candidates[np.argsort(relevances)[::-1]]))

    print()
    print()

#%%
from sklearn.cluster import KMeans
#%%
n_clusters = 5
for sofar in ['the service was ', 'the food was ', 'i loved the ', 'i ', 'the ', "we could "]:
    print("Context:", sofar)
    toks = suggestion_generator.tokenize_sofar(sofar)
    state = model.get_state(toks)[0]
    next_words, logprobs = model.next_word_logprobs_raw(state, toks[-1])
    vecs_for_words = all_vecs[next_words]
    vecs_for_clustering = vecs_for_words[logprobs > -5]# fixme if that leaves too few?
    print(len(vecs_for_clustering))
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100, n_init=10).fit(vecs_for_clustering)
    cluster_assignment = kmeans.predict(vecs_for_words)
    relevance = logprobs# - .5 * model.unigram_probs[next_words]
    print("Next-word clusters:")
    for cluster in range(n_clusters):
        members = np.flatnonzero(cluster_assignment == cluster)
        relevances = relevance[members]
        new_order = np.argsort(relevances)[::-1][:10]
        members = members[new_order]
        relevances = relevance[members]
        print(cluster, ', '.join('{}[{:.2f}]'.format(model.id2str[next_words[idx]], relevance[idx]) for idx in members))

    print()

    # Pretend we start typing a word.
    prefix = "g"
    matches = np.flatnonzero(np.array([model.id2str[idx].startswith(prefix) for idx in next_words]))
    cur_word_idx = matches[np.argmax(logprobs[matches])]
    print(f"Pretend we started typing {prefix!r} and the system guesses {model.id2str[next_words[cur_word_idx]]!r}")
    sims = pairwise.cosine_similarity(vecs_for_words[cur_word_idx][None, :], vecs_for_words)[0]
    candidates = np.argsort(sims)[-10:][::-1]
    relevances = relevance[candidates]
    print("So we suggest alternatives:", ', '.join('{}[{:.2f}]'.format(model.id2str[next_words[idx]], relevance[idx]) for idx in candidates[np.argsort(relevances)[::-1]]))

#    print(', '.join(model.id2str[next_words[idx]] for idx in np.argsort(sims[0])[-10:][::-1]))

    print()
    print()
