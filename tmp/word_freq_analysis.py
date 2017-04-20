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
np.random.seed(0)
train_frac = 1 - .05 - .05
num_docs = len(reviews)
indices = np.random.permutation(num_docs)
splits = (np.cumsum([train_frac, .05]) * num_docs).astype(int)
segment_indices = np.split(indices, splits)
names = ['train', 'valid', 'test']
for name, indices in zip(names, segment_indices):
    indicator = np.zeros(len(reviews), dtype=bool)
    indicator[indices] = True
    reviews[f'is_{name}'] = indicator
#%%

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
#%%
mean_llk_first5 = pd.Series([np.nanmean([log_freqs[indices[:5]] for indices in sent_indices if len(indices) > 5]) if len(sent_indices) else None for sent_indices in doc_sentence_indices])
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
#plt.savefig('figures/mean_min_unigram_llk_2.pdf')

#%%
to_plot = mean_mean_llk.dropna()
clip = np.percentile(to_plot, [2.5, 97.5])
sns.kdeplot(to_plot[yelp_is_best], clip=clip, label='Yelp best')
sns.kdeplot(to_plot[~yelp_is_best].dropna(), clip=clip, label='Yelp rest')
plt.xlabel("Mean mean unigram log likelihood")
#plt.savefig('figures/mean_mean_unigram_llk_2.pdf')

#%%
to_plot = pd.Series(mean_llk_first5).dropna()
clip = np.percentile(to_plot, [2.5, 97.5])
sns.kdeplot(to_plot[yelp_is_best], clip=clip, label='Yelp best')
sns.kdeplot(to_plot[~yelp_is_best].dropna(), clip=clip, label='Yelp rest')
plt.xlabel("Mean mean unigram log likelihood, first 5 words")
#plt.savefig('figures/mean_mean_unigram_llk_2.pdf')

#%%
from suggestion import clustering
#%%
cnnb = clustering.ConceptNetNumberBatch.load()
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5, max_df=.5, stop_words='english')
all_vecs = vectorizer.fit_transform(reviews.tokenized)
#%%
import wordfreq
sklearn_vocab = vectorizer.get_feature_names()
def get_or_zero(cnnb, item):
    try:
        return cnnb[item]
    except KeyError:
        return np.zeros(cnnb.ndim)
cnnb_vecs_for_sklearn_vocab = np.array([get_or_zero(cnnb, word) for word in sklearn_vocab])
wordfreqs_for_sklearn_vocab = [wordfreq.word_frequency(word, 'en', 'large', minimum=1e-9) for word in sklearn_vocab]
projection_mat = -np.log(wordfreqs_for_sklearn_vocab)[:,None] * cnnb_vecs_for_sklearn_vocab
#%%
from scipy.spatial.distance import pdist
#%%
def smooth_ecdf(x, samples):
    return np.interp(samples, np.sort(x), np.linspace(0,1,len(x)))
#%%
import tqdm
#%%
len_chars = reviews.text.str.len()
reviews['len_chars'] = len_chars
#%%
mean_len_chars = np.mean(len_chars)
samples = np.linspace(0, 50, 500)
ecdfs = np.empty((len(reviews), len(samples)))
for doc_idx in tqdm.tqdm(range(len(reviews))):
    words = all_vecs[doc_idx]
    if len(words.indices) < 2:
        ecdfs[doc_idx] = np.nan
    else:
        dists = pdist(words.data[:,None] * projection_mat[words.indices]) / (len_chars[doc_idx] / mean_len_chars) ** 2
        ecdfs[doc_idx] = smooth_ecdf(dists, samples)

#%%
prototypical_ecdf_all = np.nanmean(ecdfs, axis=0)
#ks_stats = np.max(np.abs(ecdfs - prototypical_ecdf[None,:]), axis=1)
#%%
prototypical_ecdf_best = np.nanmean(ecdfs[np.flatnonzero(yelp_is_best & ~reviews.is_train)], axis=0)
ks_stats = np.max(np.abs(ecdfs - prototypical_ecdf_best[None,:]), axis=1)
#%%
plt.plot(samples, prototypical_ecdf_all, label="All")
#plt.plot(samples, np.nanmean(ecdfs[np.flatnonzero(~yelp_is_best & ~reviews.is_train)], axis=0), label="Rest non-train")
#plt.plot(samples, np.nanmean(ecdfs[np.flatnonzero(~yelp_is_best & reviews.is_train)], axis=0), label="Rest train")
plt.plot(samples, np.nanmean(ecdfs[np.flatnonzero(yelp_is_best & ~reviews.is_train)], axis=0), label="Best non-train")
plt.plot(samples, np.nanmean(ecdfs[np.flatnonzero(yelp_is_best & reviews.is_train)], axis=0), label="Best train")
plt.legend(loc='best')
#%%
train_is_best = yelp_is_best & reviews.is_train
train_is_rest = ~yelp_is_best & reviews.is_train

#%%
to_plot = pd.Series(ks_stats).dropna()
clip = np.percentile(to_plot, [2.5, 97.5])
sns.kdeplot(to_plot[train_is_best], clip=clip, label='Yelp best')
sns.kdeplot(to_plot[train_is_rest].dropna(), clip=clip, label='Yelp rest')
plt.xlabel("K-S statistic to prototypical distribution (from 10% sample) of pairwise word-vec distances (on remaining 90%)")
#plt.savefig("prototypicality.pdf")
#%%
reviews['atypicality'] = ks_stats
#%%
reviews.to_csv('yelp_with_prototypicality_norm1.csv')
#%% Pretend to retype the best reviews, look at suggestions.
from suggestion import suggestion_generator
DEFAULT_CONFIG = dict(domain='yelp_train', temperature=0., rare_word_bonus=0.)
conditions = dict(
    phrase=dict(
      use_sufarr=False,
      temperature=0.,
      use_bos_suggs=False,
    ),
#    doublyRarePhrase=dict(
#      use_sufarr=True,
#      rare_word_bonus=2.,
#      use_bos_suggs=False,
#      ),
    rarePhrase=dict(
      use_sufarr=True,
      rare_word_bonus=1.,
      use_bos_suggs=False,
      )

)
conditions= {k: dict(DEFAULT_CONFIG, **v) for k, v in conditions.items()}

#%%
rs = np.random.RandomState(0)
samples = []
good_reviews = reviews[yelp_is_best & (~reviews.is_train)]
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
true_lengths = [len(sample['true_follows'].split()) for sample in samples]
true_lengths_char = [len(sample['true_follows']) for sample in samples]
sugg_lengths = [np.mean([len(sugg.split()) for sugg in sample['suggs'].split('\n')]) for sample in samples]
sugg_lengths_char = [np.mean([len(sugg) for sugg in sample['suggs'].split('\n')]) for sample in samples]
#%%
plt.scatter(true_lengths_char, sugg_lengths_char)

#%%
freq_diffs = pd.DataFrame(samples).dropna()
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
def show_sugg_contrast(group):
    meta = group[0]
    context = meta['context']
    true_follows = meta['true_follows']
    alternatives = [(sample['cond'], sample['suggs'].split('\n'))
        for sample in sorted(group, key=lambda x: x['cond'])]
    alternatives.append(('actual', [true_follows]))
    print(r'\multicolumn{3}{l}{...'+context[-30:]+r'}\\')
    print('\\\\\n'.join(r'\textbf{{{}}} & {}'.format(cond, ' & '.join(suggs)) for cond, suggs in alternatives), r'\\')

groups = list(cytoolz.partition_all(len(conditions), samples))

for i in [0,1,3,8]:
    show_sugg_contrast(groups[i])



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