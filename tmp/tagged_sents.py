import os
import pandas as pd
import numpy as np
#import nltk
import cytoolz
#%%
from suggestion import clustering
from scipy.misc import logsumexp
#%%
from suggestion.paths import paths
clizer = pd.read_pickle(os.path.join(paths.parent, 'models', 'goal_oriented_suggestion_data.pkl'))
#%%
data = pd.read_pickle(os.path.join(paths.parent, 'yelp_preproc/all_data.pkl'))
reviews = data['data'].reset_index(drop=True)

#%%
def get_topic_distribution(clizer, target_dist, sent_cluster_distribs, new_dists_opts):
    from scipy.special import kl_div
    with_new_dist = np.array([np.concatenate((sent_cluster_distribs, new_dist_opt[None]), axis=0) for new_dist_opt in new_dists_opts])
    dist_with_new_dist = clustering.normalize_dists(np.mean(with_new_dist, axis=1))
    return kl_div(dist_with_new_dist, target_dist).sum(axis=1)

#%%
if False:
    scores_by_cluster = clizer.scores_by_cluster.copy()
    likelihood_bias = logsumexp(scores_by_cluster, axis=1, keepdims=True)
    scores_by_cluster -= .85 * likelihood_bias
    scores_by_cluster[suggested_already] = -np.inf
    scores_by_cluster[clizer.omit] = -np.inf
    most_distinctive = np.argmax(scores_by_cluster, axis=0)
#%%
topic_tags = [f'<T{i}>' for i in range(10)]

#%%

# FIXME: this approach is super-biased for predicting topic tags because the topic tags repeat for every sentence.
# Better would be to separately train topics.
def review_to_tagged_sents(sents):
    cluster_distances = cytoolz.thread_first(
        sents,
        clizer.vectorize_sents,
        clustering.normalize_vecs,
        clizer.clusterer.transform)
    clusters_for_sents = np.argmin(cluster_distances, axis=1)

    res = []
    for i, sent in enumerate(sents):
        res.append([topic_tags[c] for c in clusters_for_sents[:i+1][-4:]] + sent.lower().split())
    return res

import tqdm
from suggestion import util
util.dump_kenlm('yelp_topic_tagged', [
        ' '.join(s)
        for tokenized in tqdm.tqdm(reviews.tokenized)
        for s in review_to_tagged_sents(tokenized.split('\n'))])
#%%
from suggestion import lang_model
topic2sentence_lm = lang_model.Model.from_basename(paths.model_basename('yelp_topic_tagged'))
#%%
import itertools
topic_transitions_indices = list(itertools.product(range(10), range(10)))
rev_topic_transitions_indices = [10*i+i for i in range(10)]
#%%
transition_log_likelihoods = np.array([[topic2sentence_lm.score_seq(topic2sentence_lm.get_state([topic_tags[c1], topic_tags[c2]], bos=True)[0], k)[0] for c1, c2 in itertools.product(range(10), range(10))] for k in tqdm.tqdm(clizer.unique_starts, desc="Score starts")])
#%%
#scores_by_cluster = scores_by_cluster_raw.copy()
#likelihood_bias = logsumexp(scores_by_cluster, axis=1, keepdims=True)
#%%
#unconditional_likelihood_bias = np.array([[topic2sentence_lm.score_seq(topic2sentence_lm.get_state([topic_tags[c]], bos=True)[0], k)[0] for c in range(10)] for k in tqdm.tqdm(clizer.unique_starts, desc="Score starts")])
unconditional_likelihood_bias_2 = np.array([
        logsumexp(scores_by_cluster_raw[:,10*i:10*(i+1)], axis=1) for i in range(10)]).T
#%%
scores_by_cluster = transition_log_likelihoods - .9*logsumexp(transition_log_likelihoods, axis=1, keepdims=True)#[:,rev_topic_transitions_indices] - 1. * unconditional_likelihood_bias_2
scores_by_cluster = scores_by_cluster[:,rev_topic_transitions_indices]
for cluster_idx in range(clizer.n_clusters):
    i = cluster_idx# + cluster_idx*10
#    print(topic_transitions_indices[i])
    print(i)
    for idx in np.argsort(scores_by_cluster[:,i])[-5:][::-1]:
        print(' '.join(clizer.unique_starts[idx]))
    print('\n\n')
#%%
for i in np.argsort(scores_by_cluster[:,8])[-10:]: print(' '.join(clizer.unique_starts[i]))
#%%
np.save('topic_continuation_scores.npy', scores_by_cluster)

#%%
#%%
def get_topic_seq(sents):
    cluster_distances = cytoolz.thread_first(
        sents,
        clizer.vectorize_sents,
        clustering.normalize_vecs,
        clizer.clusterer.transform)
    return np.argmin(cluster_distances, axis=1)
topic_seqs =  [get_topic_seq(tokenized.split('\n')) for tokenized in tqdm.tqdm(reviews.tokenized)]
#%%
# TODO: This actually needs to pass --discount_fallback to lmplz.
util.dump_kenlm('yelp_topic_seqs', [' '.join(topic_tags[c] for c in seq) for seq in topic_seqs])
#%%
def review_to_tagged_sents(topic_seq, sents):
    res = []
    for i, sent in enumerate(sents):
        res.append([topic_tags[c] for c in topic_seq[:i+1][-4:]] + sent.lower().split())
    return res
