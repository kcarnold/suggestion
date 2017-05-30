import os
import pandas as pd
import numpy as np
#import nltk
import cytoolz
#%%
from suggestion import clustering
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

def normal_lik(x, sigma):
    return np.exp(-.5*(x/sigma)**2) / (2*np.pi*sigma)

def likelihoods_by_sentence(sents):
    sent_cluster_distribs = cytoolz.thread_first(
        sents,
        clizer.vectorize_sents,
        clustering.normalize_vecs,
        clizer.clusterer.transform,
        (normal_lik, .5),
        clustering.normalize_dists
        )
    hard_assignments = np.argmax(sent_cluster_distribs, axis=1)
#    print(hard_assignments)
    return [
        assignment in np.argsort(get_topic_distribution(
            clizer=clizer,
            target_dist=clizer.target_dists['best'],
            sent_cluster_distribs=sent_cluster_distribs[:i],
            new_dists_opts=np.eye(clizer.n_clusters)))[:3]
        for i, assignment in enumerate(hard_assignments)]
def frac_suggestions_accepted(tokenized):
    return np.mean(likelihoods_by_sentence(tokenized.split('\n')))
#%%
reviews['frac_suggestions_accepted'] = reviews.tokenized.apply(frac_suggestions_accepted)
#frac_suggestions_accepted(reviews.tokenized.iloc[1])
#%%
likelihoods_by_sentence(reviews.tokenized.iloc[0].split('\n'))
#%%
reviews.to_csv('reviews_with_frac_suggs_accepted.csv')
