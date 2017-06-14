# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:04:10 2017

@author: kcarnold
"""

from suggestion import train_ngram
import pandas as pd
import numpy as np
from suggestion.paths import paths
#%%
reviews = pd.read_pickle(str(paths.parent / 'yelp_preproc' / 'valid_data.pkl'))['data']
#%%
items = [sent for tokenized in reviews.tokenized for sent in tokenized.lower().split('\n')]
doc_indices = [doc_idx for doc_idx, tokenized in enumerate(reviews.tokenized) for sent in tokenized.split('\n')]
labels = np.array(reviews.stars_review.iloc[doc_indices])
#%%
items_initial = [sent.split()[:5] for sent in items]
long_enough = np.flatnonzero((np.array([len(sent) for sent in items_initial]) == 5) & (labels != 3))
items_initial = [items_initial[i] for i in long_enough]
labels_initial = np.array([labels[i] for i in long_enough])
#%%
# Evaluate the LM classifier
from suggestion import suggestion_generator
lm_clf = suggestion_generator.sentiment_classifier
#%%
bos_state = lm_clf.get_state(['<S>'], bos=True)
#%%
import tqdm
posteriors = np.array([np.mean(lm_clf.classify_seq_by_tok(bos_state, seq), axis=0) for seq in tqdm.tqdm(items_initial)])
#%%
# Let's turn this into a binary classification.
is_positive = labels_initial > 3
pos_probs = posteriors[:,3] + posteriors[:,4]
neg_probs = posteriors[:,0] + posteriors[:,1]
pos_probs = pos_probs / (pos_probs + neg_probs)
#%%
from sklearn.metrics import roc_auc_score
roc_auc_score(is_positive, pos_probs)
# -> 0.72
#%% Compare this to a simple NB classifier.
training_reviews = pd.read_pickle(str(paths.parent / 'yelp_preproc' / 'train_data.pkl'))['data'].query('stars_review != 3')
#%%
training_items = [sent for tokenized in training_reviews.tokenized for sent in tokenized.lower().split('\n')]
training_doc_indices = [doc_idx for doc_idx, tokenized in enumerate(training_reviews.tokenized) for sent in tokenized.split('\n')]
training_labels = np.array(training_reviews.stars_review.iloc[training_doc_indices])
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
vecs = vectorizer.fit_transform(training_items)
#%%
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB().fit(vecs, training_labels > 3)
roc_auc_score(is_positive, clf.predict_proba(vectorizer.transform([' '.join(seq) for seq in items_initial]))[:,1])
# -> 0.759
# Hm, that works better doesn't it...
# for unigrams only, we get 0.72. So bigrams help.
#%%
np.mean(is_positive == clf.predict(vectorizer.transform([' '.join(seq) for seq in items_initial])))
# -> 0.7624046671325696
#%%
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(vecs, training_labels > 3)
roc_auc_score(is_positive, clf.predict_proba(vectorizer.transform([' '.join(seq) for seq in items_initial]))[:,1])
# -> 0.754
#%%
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
clf.fit(vecs, training_labels > 3)
roc_auc_score(is_positive, clf.predict_proba(vectorizer.transform([' '.join(seq) for seq in items_initial]))[:,1])
# -> 0.676.
#%%
# Try training smaller models.
from suggestion import util
for stars in tqdm.trange(1, 6):
    util.make_model(f'yelp_train-{stars}star', order=2)
#%%
reload(suggestion_generator)
lm_clf = suggestion_generator.sentiment_classifier
#%%
bos_state = lm_clf.get_state(['<S>'], bos=True)
posteriors = np.array([np.mean(lm_clf.classify_seq_by_tok(bos_state, seq), axis=0) for seq in tqdm.tqdm(items_initial)])
#%%
is_positive = labels_initial > 3
pos_probs = posteriors[:,3] + posteriors[:,4]
neg_probs = posteriors[:,0] + posteriors[:,1]
pos_probs = pos_probs / (pos_probs + neg_probs)
#%%
from sklearn.metrics import roc_auc_score
roc_auc_score(is_positive, pos_probs)
# can't train an order-1 using KenLM :(
# order=2 -> 0.736. Slightly better than order=5. Not quite as good as using the full NB classifier.
# order=3 -> 0.733. Going down.
# order=5 -> 0.72.

#%% I wonder if the benefit still holds mid-sentence.
contexts = []
suffixes = []
labels_mid = []
rs = np.random.RandomState(1)
for sent, label in zip(tqdm.tqdm(items), labels):
    sent = sent.split()
    if len(sent) <= 6:
        continue
    # Ensure that the context has at least one token and that there are >= 5 tokens left in the sentence.
    split_idx = rs.randint(1, len(sent) - 5)
    contexts.append(sent[:split_idx])
    suffix = sent[split_idx:]
    assert len(suffix) >= 5
    suffixes.append(suffix[:5])
    labels_mid.append(label)
labels_mid = np.array(labels_mid)
#%%
posteriors = np.array([np.mean(lm_clf.classify_seq_by_tok(lm_clf.get_state(['<S>'] + context), seq), axis=0) for context, seq in zip(tqdm.tqdm(contexts), suffixes)])
#%%
# Turn this into a binary classification.
is_positive = labels_mid > 3
pos_probs = posteriors[:,3] + posteriors[:,4]
neg_probs = posteriors[:,0] + posteriors[:,1]
pos_probs = pos_probs / (pos_probs + neg_probs)
roc_auc_score(is_positive, pos_probs)
# -> 0.653 (for order=2, on two separate splitting trials)
# -> 0.650 (for order=3)
# -> 0.646 for order=5
#%% vs the BernoulliNB
vectorizer = TfidfVectorizer(ngram_range=(1,2))
vecs = vectorizer.fit_transform(training_items)
clf = BernoulliNB().fit(vecs, training_labels > 3)
#%%
roc_auc_score(is_positive, clf.predict_proba(vectorizer.transform([' '.join(seq) for seq in suffixes]))[:,1])
# -> 0.67



#%%
# How about the full sentiment classification task?
roc_auc_score(np.eye(5)[labels_mid-1], posteriors, average='weighted')
# -> 0.62 with average='macro'
# -> 0.604 with average='weighted'
# -> 0.584 for order=5
#%%
# Train on full review data
training_reviews = pd.read_pickle(str(paths.parent / 'yelp_preproc' / 'train_data.pkl'))['data']
training_items = [sent for tokenized in training_reviews.tokenized for sent in tokenized.lower().split('\n')]
training_doc_indices = [doc_idx for doc_idx, tokenized in enumerate(training_reviews.tokenized) for sent in tokenized.split('\n')]
training_labels = np.array(training_reviews.stars_review.iloc[training_doc_indices])
#%%
vectorizer = TfidfVectorizer(ngram_range=(1,2))
vecs = vectorizer.fit_transform(training_items)
#%%
clf = BernoulliNB().fit(vecs, training_labels)
#%%
roc_auc_score(np.eye(5)[labels_mid-1], clf.predict_proba(vectorizer.transform([' '.join(seq) for seq in suffixes])), average='weighted')
# -> 0.63 for average='macro'
# -> 0.604 for average='weighted'

