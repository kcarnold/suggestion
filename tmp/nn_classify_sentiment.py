# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:38:15 2017

@author: kcarnold
"""
import numpy as np
#%%

from suggestion.analyzers import load_reviews
reviews = load_reviews()
#%%
from collections import Counter
import itertools
import tqdm
vocab = Counter(itertools.chain.from_iterable(text.lower().split() for text in tqdm.tqdm(reviews[reviews.is_train].tokenized)))
#%%
MAX_SEQ_LEN=100
#%%
NUM_WORDS = 20000
id2str = ['<PAD>', '<UNK>'] + [word for word, count in vocab.most_common(NUM_WORDS)]
str2id = {word: idx for idx, word in enumerate(id2str)}
#%%
from suggestion import clustering
cnnb = clustering.ConceptNetNumberBatch.load()
#%%
EMBEDDING_DIM = 300
num_words = len(id2str)
embedding_mat = np.zeros((num_words, EMBEDDING_DIM))
#%%
#from sklearn.random_projection import GaussianRandomProjection

#%%
random_state = np.random.RandomState(0)

def random_vec(ndim, rs):
    vec = rs.standard_normal(ndim)
    vec /= np.linalg.norm(vec)
    return vec

num_words_random = 0
for word_idx, word in enumerate(tqdm.tqdm(id2str, desc="Loading embeddings")):
    try:
        embedding_mat[word_idx] = cnnb[word]
    except KeyError:
        num_words_random += 1
        embedding_mat[word_idx] = random_vec(EMBEDDING_DIM, random_state)
#%%
def to_seqs(newline_tokenized, doc_level_labels, str2id, unk_id):
#    num_sents = sum(1 for doc in newline_tokenized for sent in doc.split('\n'))
#    res = np.empty((num_sents, seq_len))
    res_x = []
    res_y = []
    for doc, label in zip(newline_tokenized, doc_level_labels):
        for sent in doc.split('\n'):
            sent = sent.split()
            if len(sent) == 0:
                continue
            res_x.append([str2id.get(tok, unk_id) for tok in sent])
            res_y.append(label)
    return res_x, res_y
x_train_fullseq, y_train = to_seqs(reviews[reviews.is_train].tokenized, reviews[reviews.is_train].stars_review, str2id, str2id['<UNK>'])
#%%
def random_subset_seqs(seqs, subseq_len, *, pad_val=0., random_state):
    X = np.full((len(seqs), subseq_len), pad_val)
    for i, seq in enumerate(seqs):
        start_idx = random_state.choice(len(seq))
        end_idx = min(start_idx + subseq_len, len(seq))
        this_subseq_len = end_idx - start_idx
        X[i, :this_subseq_len] = seq[start_idx:end_idx]
    return X
x_train = random_subset_seqs(x_train_fullseq, 10, random_state=random_state)
#%%
x_test_fullseq, y_test = to_seqs(reviews[reviews.is_valid].tokenized, reviews[reviews.is_valid].stars_review, str2id, str2id['<UNK>'])
x_test = random_subset_seqs(x_test_fullseq, 10, random_state=random_state)
#%%
y_train = np.array(y_train) - 1
y_test = np.array(y_test) - 1
#%%
num_labels = 5
#%%

#%%
import joblib
joblib.dump(dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, id2str=id2str, embedding_mat=embedding_mat), 'keras_training_data.pkl')
#%%
