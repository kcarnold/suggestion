# -*- coding: utf-8 -*-
import cytoolz
import numpy as np
import joblib
import attr
import wordfreq
import pickle
from suggestion import suggestion_generator, paths
from suggestion.util import dump_kenlm, mem
import tqdm
from scipy.misc import logsumexp
from scipy.spatial.distance import cdist
import logging
logger = logging.getLogger(__name__)

JOBLIB_FILENAME = '/Data/conceptnet-vector-ensemble/conceptnet-numberbatch-201609-en.joblib'

@attr.s
class ConceptNetNumberBatch:
    term2id = attr.ib()
    id2term = attr.ib()
    vecs = attr.ib()
    ndim = attr.ib()

    @staticmethod
    def extract_english(h5_filename='conceptnet-numberbatch-201609.h5'):
        import h5py
        f = h5py.File(h5_filename)
        labels = f['mat']['axis1'].value
        en_labels = [lbl[6:].decode('utf8') for idx, lbl in enumerate(labels) if lbl.startswith(b'/c/en/')]
        en_indices = [idx for idx, lbl in enumerate(labels) if lbl.startswith(b'/c/en/')]
        en_indices = [idx for idx, lbl in enumerate(labels) if lbl.startswith(b'/c/en/')]
        en_vecs = f['mat']['block0_values'][en_indices]
        return dict(labels=en_labels, vecs=en_vecs)

    @classmethod
    def save_joblib(cls):
        joblib.dump(cls.extract_english(), JOBLIB_FILENAME)

    @classmethod
    def load(cls):
        data = joblib.load(JOBLIB_FILENAME, mmap_mode='r')
        id2term = data['labels']
        term2id = {term: idx for idx, term in enumerate(id2term)}
        vecs = data['vecs']
        return cls(vecs=vecs, term2id=term2id, id2term=id2term, ndim=vecs.shape[1])

    def __getitem__(self, item):
        return self.vecs[self.term2id[item]]

    def __contains__(self, item):
        return item in self.term2id

cnnb = ConceptNetNumberBatch.load()

def get_all_sents():
    data = pickle.load(open('yelp_preproc/all_data.pkl', 'rb'))
    reviews = data['data'].reset_index(drop=True)
    return list(cytoolz.concat(doc.lower().split('\n') for doc in reviews.tokenized))

def filter_reasonable_length_sents(sents):
    sent_lens = np.array([len(sent.split()) for sent in sents])
    min_sent_len, max_sent_len = np.percentile(sent_lens, [25, 75])
    return [sent for sent in sents if min_sent_len <= len(sent.split()) <= max_sent_len]

@mem.cache
def get_vectorizer(sents):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=5, max_df=.5, stop_words='english')
    all_vecs = vectorizer.fit_transform(sents)
    return vectorizer, all_vecs

@mem.cache
def get_projection_mat(vectorizer):
    sklearn_vocab = vectorizer.get_feature_names()
    def get_or_zero(cnnb, item):
        try:
            return cnnb[item]
        except KeyError:
            return np.zeros(cnnb.ndim)
    cnnb_vecs_for_sklearn_vocab = np.array([get_or_zero(cnnb, word) for word in sklearn_vocab])
    wordfreqs_for_sklearn_vocab = [wordfreq.word_frequency(word, 'en', 'large', minimum=1e-9) for word in sklearn_vocab]
    return -np.log(wordfreqs_for_sklearn_vocab)[:,None] * cnnb_vecs_for_sklearn_vocab

def filter_by_norm(vecs, texts, min_norm=.5):
    norms = np.linalg.norm(vecs, axis=1)
    large_enough = norms > min_norm
    vecs = vecs[large_enough] / norms[large_enough][:,None]
    texts = [texts[i] for i in np.flatnonzero(large_enough)]
    return texts, vecs

@mem.cache
def get_clusterer_and_dists(docs_projected, n_clusters, random_state):
    from sklearn.cluster import MiniBatchKMeans
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=random_state)
    cluster_dists = mbk.fit_transform(docs_projected)
    return mbk, cluster_dists

def summarize_clusters(doc_texts, cluster_dists):
    for c in range(cluster_dists.shape[1]):
        print(c)
        for i in np.argsort(cluster_dists[:,c])[:10]:
            print(i, doc_texts[i].replace('\n', ' '))
        print()

def train_models_per_cluster(mbk, vecs, texts):
    sentences_in_cluster = [[] for i in range(mbk.n_clusters)]
    for i, c in enumerate(mbk.predict(vecs)):
        sentences_in_cluster[c].append(texts[i])
    for cluster_idx, cluster in enumerate(sentences_in_cluster):
        print(cluster_idx)
        dump_kenlm('cluster_{}'.format(cluster_idx), [s.lower() for s in cluster])


def normalize_vecs(vecs):
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-7)

def normalize_dists(dists):
    return dists / np.sum(dists, axis=1, keepdims=True)


class Clusterizer:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters

        logger.info("Loading sentences")
        sents = filter_reasonable_length_sents(get_all_sents())

        logger.info("Vectorizing")
        self.vectorizer, raw_vecs = get_vectorizer(sents)
        self.projection_mat = get_projection_mat(self.vectorizer)
        vecs = raw_vecs.dot(self.projection_mat)
        self.sents, self.vecs = filter_by_norm(vecs, sents)
        del vecs

        logger.info("Clustering")
        self.clusterer, cluster_dists = get_clusterer_and_dists(self.vecs, n_clusters=n_clusters, random_state=0)

        logger.info("Training sub-models")
        train_models_per_cluster(self.clusterer, vecs=self.vecs, texts=self.sents)

        self.models = [suggestion_generator.Model.from_basename(paths.paths.model_basename('cluster_{}'.format(cluster_idx))) for cluster_idx in range(n_clusters)]

        #%% Score the first 5 words of every sentence.
        self.unique_starts = [x.split() for x in sorted({' '.join(sent.split()[:5]) for sent in self.sents})]
        self.scores_by_cluster = np.array([[model.score_seq(model.bos_state, k)[0] for model in self.models] for k in tqdm.tqdm(self.unique_starts, desc="Score starts")])

    def get_distinctive():
        sbc_scale = .25 * scores_by_cluster# + 1*scores[:,None] - 1 * unigram_llks_for_start[:,None]
        likelihood_bias = logsumexp(sbc_scale, axis=1, keepdims=True)
        scores_by_cluster_debias = sbc_scale - .5*likelihood_bias# - 0*scores[:,None]
        most_distinctive = np.argmax(scores_by_cluster_debias, axis=0)
        for cluster, sent_idx in enumerate(most_distinctive):
            print('{:4.2f} {}'.format(np.exp(scores_by_cluster_debias[sent_idx, cluster]), ' '.join(unique_starts[sent_idx])))
        #print('\n'.join([) for i in most_distinctive]))

    def vectorize_sents(self, sents):
        return self.vectorizer.transform(sents).dot(self.projection_mat)
