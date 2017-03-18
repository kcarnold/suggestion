# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:47:36 2017

@author: kcarnold
"""

import numpy as np
import joblib
import attr
#%%
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
#%%

topic_words_data = {
    'food': '''
food egg eggs omelet burrito wrap taste tasted tastes salad salads fresh greasy knife fork spoon filling tasty edible fluffy tender delicious
fries shrimp salmon grits duck cook hummus tahini falafel meat sandwich sandwiches dishes ingredients steak peppers onions
    ''',
    'ambiance': '''
    looked window windows atmosphere ambiance cramped outside packed dark dirty loud clean cleaner quiet quieter view
    ''',
    'value': '''
    price prices priced pricey portions deal spend fill cheap charged
    ''',
    'service': '''
    quick quickly service slow fast quickly pleasant cashier waiter waiters host hostess
    ''',
    'drinks': '''
    beer beers coffee drink drinks tea milk
    ''',
    'desert': '''desert cake pie
    '''}
topic_words = {topic: [w.strip() for w in words.split()] for topic, words in topic_words_data.items()}
topic_vecs = {topic: np.mean([cnnb[word] for word in words], axis=0) for topic, words in topic_words.items()}
#{topic: np.linalg.norm(vec) for topic, vec in topic_vecs.items()}
#%%

def show_sim(cnnb, vec):
    sims = np.dot(cnnb.vecs, vec)
    return [cnnb.id2term[i] for i in np.argsort(sims)[::-1][:100]]

for topic, vec in topic_vecs.items():
    print(topic)
    print(' '.join(show_sim(cnnb, vec)))
    print()
#%%
import wordfreq
def vectorize(sent):
    toks = wordfreq.tokenize(sent, 'en')
    tot_vec = np.zeros(cnnb.ndim)
    tot_weight = 0.
    for tok in toks:
        tok = tok.lower()
        try:
            vec = cnnb[tok]
            weight = -np.log(wordfreq.word_frequency(tok, 'en', 'large', minimum=1e-9))
            tot_weight += weight
            tot_vec += vec
        except IndexError:
            pass
    if tot_weight > 0:
        return tot_vec / tot_weight
    return None

vectorize('We came here on a Friday night')
#%%

import pandas as pd
import pickle
import numpy as np
#%% Load all the reviews.
data = pickle.load(open('yelp_preproc/all_data.pkl','rb'))
vocab, counts = data['vocab']
reviews = data['data'].reset_index(drop=True)
del data
#%%
import cytoolz
sents = list(cytoolz.concat(doc.split('\n') for doc in reviews.tokenized))
#%%
sent_lens = np.array([len(sent.split()) for sent in sents])
min_sent_len, max_sent_len = np.percentile(sent_lens, [25, 75])
reasonable_length_sents = [sent for sent in sents if min_sent_len <= len(sent.split()) <= max_sent_len]
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5, max_df=.5, stop_words='english')
all_vecs = vectorizer.fit_transform(reasonable_length_sents)
#%%
sklearn_vocab = vectorizer.get_feature_names()
def get_or_zero(cnnb, item):
    try:
        return cnnb[item]
    except KeyError:
        return np.zeros(cnnb.ndim)
cnnb_vecs_for_sklearn_vocab = np.array([get_or_zero(cnnb, word) for word in sklearn_vocab])
#%%
wordfreqs_for_sklearn_vocab = [wordfreq.word_frequency(word, 'en', 'large', minimum=1e-9) for word in sklearn_vocab]
weighted_cnnb_vecs = -np.log(wordfreqs_for_sklearn_vocab)[:,None] * cnnb_vecs_for_sklearn_vocab
#%%
all_docs_projected = all_vecs.dot(weighted_cnnb_vecs)
#%%
doc_norms = np.linalg.norm(all_docs_projected, axis=1)
large_enough = doc_norms > .5
docs_projected = all_docs_projected[large_enough] / doc_norms[large_enough][:,None]
doc_texts = [reasonable_length_sents[i] for i in np.flatnonzero(large_enough)]
#%%
#%%
from sklearn.cluster import MiniBatchKMeans
mbk = MiniBatchKMeans(init='k-means++', n_clusters=10, n_init=10)
cluster_dists = mbk.fit_transform(docs_projected)
#%%
for c in range(cluster_dists.shape[1]):
    print(c)
    for i in np.argsort(cluster_dists[:,c])[:10]:
        print(i, doc_texts[i].replace('\n', ' '))
    print()
#%%
import subprocess
def dump_kenlm(model_name, tokenized_sentences):
    # Dump '\n'.join(' '.join-formatted tokenized reviews, without special markers,
    # to a file that KenLM can read, and build a model with it.
    with open('models/{}.txt'.format(model_name), 'w') as f:
        for toks in tokenized_sentences:
            print(toks.lower(), file=f)
    subprocess.run(['./scripts/make_model.sh', model_name])
#%%
sentences_in_cluster = [[] for i in range(mbk.n_clusters)]
for i, c in enumerate(mbk.predict(docs_projected)):
    sentences_in_cluster[c].append(doc_texts[i])
#%%
[len(c) for c in sentences_in_cluster]
#%%
for cluster_idx, cluster in enumerate(sentences_in_cluster):
    print(cluster_idx)
    dump_kenlm('cluster_{}'.format(cluster_idx), [s.lower() for s in cluster])
#%%
from suggestion import suggestion_generator, paths
models = [suggestion_generator.Model.from_basename(paths.paths.model_basename('cluster_{}'.format(cluster_idx))) for cluster_idx in range(mbk.n_clusters)]
#%% Score the first 5 words of every sentence.
unique_starts = [x.split() for x in sorted({' '.join(sent.split()[:5]) for sent in doc_texts})]
#%%
import tqdm
scores_by_cluster = np.array([[model.score_seq(model.bos_state, k)[0] for model in models] for k in tqdm.tqdm(unique_starts)])
#%%
from scipy.misc import logsumexp
sbc_scale = .25 * scores_by_cluster# + 1*scores[:,None] - 1 * unigram_llks_for_start[:,None]
likelihood_bias = logsumexp(sbc_scale, axis=1, keepdims=True)
scores_by_cluster_debias = sbc_scale - .5*likelihood_bias# - 0*scores[:,None]
most_distinctive = np.argmax(scores_by_cluster_debias, axis=0)
for cluster, sent_idx in enumerate(most_distinctive):
    print('{:4.2f} {}'.format(np.exp(scores_by_cluster_debias[sent_idx, cluster]), ' '.join(unique_starts[sent_idx])))
#print('\n'.join([) for i in most_distinctive]))