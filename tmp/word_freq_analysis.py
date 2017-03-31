# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:32:52 2017

@author: kcarnold
"""
import pandas as pd
import pickle
import numpy as np
import cytoolz

#%% Load all the reviews.
data = pickle.load(open('yelp_preproc/all_data.pkl','rb'))
vocab, counts = data['vocab']
reviews = data['data'].reset_index(drop=True)
del data
#%%
freqs = counts / counts.sum()
word2idx = {word: idx for idx, word in enumerate(vocab)}
log_freqs = np.log(freqs)
#%%
def lookup_indices(sent):
    tmp = (word2idx.get(word) for word in sent)
    return [w for w in tmp if w is not None]

def mean_log_freq(indices):
    return np.mean(log_freqs[indices]) if len(indices) else None

def min_log_freq(indices):
    return np.min(log_freqs[indices]) if len(indices) else None
#%%
doc_sentence_indices = [[lookup_indices(sent.split()) for sent in doc.lower().split('\n')] for doc in reviews.tokenized]
#%%
mean_llk = [list(cytoolz.filter(None, [mean_log_freq(indices) for indices in doc_indices])) for doc_indices in doc_sentence_indices]
min_llk = [list(cytoolz.filter(None, [min_log_freq(indices) for indices in doc_indices])) for doc_indices in doc_sentence_indices]
#%%
mean_mean_llk = pd.Series([np.mean(llks) if len(llks) > 0 else None for llks in mean_llk])
mean_min_llk = pd.Series([np.mean(llks) if len(llks) > 0 else None for llks in min_llk])

#%% Identify the best reviews.
# Mark the top reviews: top-5 ranked reviews of restaurants with at least the median # reviews,
# as long as they have >= 10 votes.
reviews['total_votes'] = reviews['votes_cool'] + reviews['votes_funny'] + reviews['votes_useful']
reviews['total_votes_rank'] = reviews.groupby('business_id').total_votes.rank(ascending=False)
business_review_counts = reviews.groupby('business_id').review_count.mean()
median_review_count = np.median(business_review_counts)
yelp_is_best = (reviews.review_count >= median_review_count) & (reviews.total_votes >= 10) & (reviews.total_votes_rank <= 5)
#%%
num_sents = np.array([len(text.split('\n')) for text in reviews.tokenized])
#%%
import seaborn as sns
import matplotlib.pyplot as plt
#%%
to_plot = mean_min_llk.dropna()
clip = np.percentile(to_plot, [2.5, 97.5])
sns.kdeplot(to_plot[yelp_is_best], clip=clip, label='Yelp best')
sns.kdeplot(to_plot[~yelp_is_best].dropna(), clip=clip, label='Yelp rest')
plt.xlabel("Mean min unigram log likelihood")
plt.savefig('figures/mean_min_unigram_llk_2.pdf')

#%%
to_plot = mean_mean_llk.dropna()
clip = np.percentile(to_plot, [2.5, 97.5])
sns.kdeplot(to_plot[yelp_is_best], clip=clip, label='Yelp best')
sns.kdeplot(to_plot[~yelp_is_best].dropna(), clip=clip, label='Yelp rest')
plt.xlabel("Mean mean unigram log likelihood")
plt.savefig('figures/mean_mean_unigram_llk_2.pdf')

#%% Pretend to retype the best reviews, look at suggestions.
from suggestion import suggestion_generator
DEFAULT_CONFIG = dict(domain='yelp_train', temperature=0., rare_word_bonus=0.)
conditions = dict(
    phrase=dict(
      use_sufarr=False,
      temperature=0.,
      use_bos_suggs=False,
    ),
    rarePhrase=dict(
      use_sufarr=True,
      rare_word_bonus=2.,
      use_bos_suggs=True, # but we won't test that case.
      ),
    halfRarePhrase=dict(
      use_sufarr=True,
      rare_word_bonus=1.,
      use_bos_suggs=True, # but we won't test that case.
      )

)
conditions= {k: dict(DEFAULT_CONFIG, **v) for k, v in conditions.items()}

#%%
rs = np.random.RandomState(0)
samples = []
good_reviews = reviews[yelp_is_best]
import tqdm
#%%
target = 10000
progress = tqdm.tqdm(total=target)
progress.update(len(samples) // 3)
while len(samples) < len(conditions)*target:
    review_idx = rs.choice(len(good_reviews))
    review = good_reviews.iloc[review_idx]
    sents = [sent for sent in review.tokenized.lower().split('\n')]
    sent_idx = rs.choice(len(sents))
    sent_toks = sents[sent_idx].split()
    sent_len = len(sent_toks)
    if sent_len < 7: # make sure we have at least one word to start, and don't count final punct
        continue
    word_idx = rs.randint(1, sent_len - 5)
    context = ' '.join(sents[:sent_idx] + sent_toks[:word_idx])
    true_follows = sent_toks[word_idx:][:5]
    for cond, flags in conditions.items():
        try:
            suggestions = [phrase for phrase, probs in suggestion_generator.get_suggestions(
                sofar=context+' ', cur_word=[], **flags)[0]]
        except Exception:
            suggestions = []
        mean_log_freq_true = mean_log_freq(lookup_indices(true_follows))
        mean_log_freq_sugg = np.nanmean([mean_log_freq(lookup_indices(sugg)) for sugg in suggestions])
        samples.append(dict(
                review_idx=review_idx,
                sent_idx=sent_idx,
                word_idx=word_idx,
                mean_log_freq_true=mean_log_freq_true,
                mean_log_freq_sugg=mean_log_freq_sugg,
                cond=cond, context=context,
                true_follows=' '.join(true_follows),
                suggs='\n'.join(' '.join(sugg) for sugg in suggestions)))
    progress.update(1)
#%%
freq_diffs = pd.DataFrame(samples)
freq_diffs['diff'] = freq_diffs['mean_log_freq_true'] - freq_diffs['mean_log_freq_sugg']
freq_diffs.groupby('cond').diff.mean()
#%%
sns.violinplot(x='diff', y='cond', data=freq_diffs)
#%%

clip = np.percentile(freq_diffs['diff'], [2.5, 97.5])
for cond, data in freq_diffs.groupby('cond'):
    sns.kdeplot(data['diff'], clip=clip, label=cond)
plt.savefig('figures/word_freq_of_suggs.pdf')

#%%
import random
acceptability_frames = []
for group in cytoolz.partition_all(len(conditions), samples):
    context = group[0]['context']
    meta = list(cytoolz.pluck(['review_idx', 'sent_idx', 'word_idx', 'true_follows'], group))
    assert len(set(meta)) == 1
    meta = list(meta[0])
    true_follows = meta.pop(-1)
    options = [('true', group[0]['true_follows'])]
    for sample in group:
        for sugg in sample['suggs'].split('\n')[:1]:
            options.append((sample['cond'], sugg))
    random.shuffle(options)
    acceptability_frames.append(dict(meta=meta, context=group[0]['context'], options=options))
import json
json.dump(acceptability_frames, open('acceptability_frames.json','w'))
#%%
def dict2arr(dct):
    arr = []
    for k, v in dct.items():
        while k >= len(arr):
            arr.append(None)
        arr[k] = v
    return arr
responses = dict2arr({int(k): dict2arr({int(k2): v2 for k2, v2 in v.items()}) for k, v in json.load(open('acceptability_results_me_1.json'))['responses'].items()})
responses
#%%
results = []
for frame, response in zip(acceptability_frames, responses):
    if response is None:
        continue
    for (cond, phrase), val in zip(frame['options'], response):
        if val is not None:
            results.append(dict(cond=cond, val=val))
results = pd.DataFrame(results)

#%%%%%%% Analyze topic distributions
from suggestion import clustering
clizer = clustering.Clusterizer()
#%%
def clusters_in_doc(doc_tokenized):
    vecs = clizer.vectorize_sents(doc_tokenized.split('\n'))
    norms = np.linalg.norm(vecs, axis=1)
    vecs = vecs[norms > .5]
    if len(vecs) == 0:
        return np.zeros(clizer.n_clusters)
    return np.bincount(clizer.clusterer.predict(vecs), minlength=clizer.n_clusters)

clusters_in_doc(reviews.tokenized.iloc[50])
#%%
import tqdm
clusters_in_all_docs = np.array([clusters_in_doc(tokenized) for tokenized in tqdm.tqdm(reviews.tokenized)])
#%%
cluster_probs = clusters_in_all_docs / (np.sum(clusters_in_all_docs, axis=1, keepdims=True) + 1e-9)
#%%

def normal_lik(x, sigma):
    return np.exp(-.5*(x/sigma)**2) / (2*np.pi*sigma)

def normalize_dists(dists):
    return dists / (np.sum(dists, axis=1, keepdims=True) + 1e-6)
#%%
def cluster_dist_in_doc(doc_tokenized):
    vecs = clizer.vectorize_sents(doc_tokenized.split('\n'))
    norms = np.linalg.norm(vecs, axis=1)
    vecs = vecs[norms > .5]
    if len(vecs) == 0:
        return np.zeros(clizer.n_clusters)
    return np.mean(normalize_dists(normal_lik(clizer.clusterer.transform(vecs), .5)), axis=0)

cluster_dist_in_all_docs = np.array([cluster_dist_in_doc(tokenized) for tokenized in tqdm.tqdm(reviews.tokenized)])
#%%


stats = dict(
     overall=np.mean(cluster_probs, axis=0),
     best=np.mean(cluster_probs[yelp_is_best], axis=0),
     rest=np.mean(cluster_probs[~yelp_is_best], axis=0))
{k: entr(v).sum() for k, v in stats.items()}
#%%
np.mean(np.sum(cluster_probs, axis=1))
#%%
from scipy.special import entr
entropies = np.sum(entr(cluster_probs), axis=1)
#%%
entropies2 = np.sum(entr(normalize_dists(cluster_dist_in_all_docs)), axis=1)
#%%
long_enough = pd.Series(num_sents > 2)# & pd.Series(num_sents < 10)
bw=.04
to_plot = pd.Series(entropies2).dropna()
to_plot = to_plot[to_plot > 0]
clip = np.percentile(to_plot, [2.5, 97.5])
sns.kdeplot(to_plot[yelp_is_best & long_enough], clip=clip, label=f'Yelp best (mean={to_plot[yelp_is_best & long_enough].mean():.2f})', bw=bw)
sns.kdeplot(to_plot[~yelp_is_best & long_enough].dropna(), clip=clip, label=f'Yelp rest (mean={to_plot[~yelp_is_best & long_enough].mean():.2f})', bw=bw)
plt.xlabel("Entropy of cluster distribution")
plt.savefig('figures/cluster_distribution_entropy.pdf')