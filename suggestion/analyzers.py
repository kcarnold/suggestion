import os
import pickle
import numpy as np
import tqdm
from scipy.spatial.distance import pdist
from .paths import paths
import pandas as pd


def load_reviews():
    data = pd.read_pickle(str(paths.preproc / 'all_data.pkl'))
    reviews = data['data'].reset_index(drop=True)

    # Reconstruct train/test split indices.
    # TODO: put this in the preprocessing
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

    ## Identify the best reviews.
    # Mark the top reviews: top-5 ranked reviews of restaurants with at least the median # reviews,
    # as long as they have >= 10 votes.
    reviews['total_votes'] = reviews['votes_cool'] + reviews['votes_funny'] + reviews['votes_useful']
    reviews['total_votes_rank'] = reviews.groupby('business_id').total_votes.rank(ascending=False)
    business_review_counts = reviews.groupby('business_id').review_count.mean()
    median_review_count = np.median(business_review_counts)
    reviews['is_best'] = (reviews.review_count >= median_review_count) & (reviews.total_votes >= 10) & (reviews.total_votes_rank <= 5)
    return reviews


class WordFreqAnalyzer:
    def __init__(self, vocab, counts):
        freqs = counts / counts.sum()
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.log_freqs = np.log(freqs)

    def lookup_indices(self, toks):
        word2idx = self.word2idx
        tmp = (word2idx.get(word) for word in toks)
        return [w for w in tmp if w is not None]

    def mean_log_freq(self, indices):
        return np.mean(self.log_freqs[indices]) if len(indices) else None

    def min_log_freq(self, indices):
        return np.min(self.log_freqs[indices]) if len(indices) else None

    def __call__(self, doc):
        '''Assumes doc is tokenized. [TODO: into sentences by newlines].'''
        indices = self.lookup_indices(doc.split()) # for sent in doc.lower().split('\n')]
#        mean_llk = list(cytoolz.filter(None, [self.mean_log_freq(indices) for indices in doc_indices]))
#        min_llk = list(cytoolz.filter(None, [self.min_log_freq(indices) for indices in doc_indices]))
        return dict(
                mean_llk=self.mean_log_freq(indices),
                min_llk=self.min_log_freq(indices))

    @classmethod
    def build(cls):
        data = pd.read_pickle(str(paths.preproc / 'all_data.pkl'))
        vocab, counts = data['vocab']
        return cls(vocab, counts)


class WordPairAnalyzer:
    def __init__(self, vectorizer, projection_mat, samples, prototypical_ecdf_best, mean_len_chars):
        self.vectorizer = vectorizer
        self.projection_mat = projection_mat
        self.samples = samples
        self.prototypical_ecdf_best = prototypical_ecdf_best
        self.mean_len_chars = mean_len_chars

    @classmethod
    def build(cls, reviews):
        from suggestion import clustering
        cnnb = clustering.ConceptNetNumberBatch.load()

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(min_df=5, max_df=.5, stop_words='english')
        all_vecs = vectorizer.fit_transform(reviews.tokenized)

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

        len_chars = reviews.text.str.len()
        mean_len_chars = np.mean(len_chars)
        samples = np.linspace(0, 50, 500)
        ecdfs = np.empty((len(reviews), len(samples)))
        for doc_idx in tqdm.tqdm(range(len(reviews))):
            words = all_vecs[doc_idx]
            if len(words.indices) < 2:
                ecdfs[doc_idx] = np.nan
            else:
                dists = pdist(words.data[:,None] * projection_mat[words.indices])# / (len_chars[doc_idx] / mean_len_chars) ** 2
                ecdfs[doc_idx] = smooth_ecdf(dists, samples)

        prototypical_ecdf_best = np.nanmean(ecdfs[np.flatnonzero(reviews.is_best & ~reviews.is_train)], axis=0)
        return cls(vectorizer, projection_mat, samples, prototypical_ecdf_best, mean_len_chars)

    def ecdf(self, doc_vectorized):
        if len(doc_vectorized.indices) < 2:
            return np.full_like(self.samples, np.nan)
        else:
            dists = pdist(doc_vectorized.data[:,None] * self.projection_mat[doc_vectorized.indices])# / (len(doc_vectorized) / mean_len_chars) ** 2
            return smooth_ecdf(dists, self.samples)

    def __call__(self, doc):
        doc_vectorized = self.vectorizer.transform([doc])
        return np.max(np.abs(self.ecdf(doc_vectorized) - self.prototypical_ecdf_best))




def smooth_ecdf(x, samples):
    return np.interp(samples, np.sort(x), np.linspace(0,1,len(x)))

