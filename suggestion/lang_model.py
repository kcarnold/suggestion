from collections import defaultdict
import numpy as np
import heapq
import sys
import string
import datrie
import nltk
import itertools
from scipy.misc import logsumexp
LOG10 = np.log(10)

import kenlm


def get_arpa_data(filename):
    with open(filename) as f:
        # read unigrams, for vocab
        while not f.readline().startswith('\\1-grams:'):
            continue
        vocab = []
        unigram_probs = []
        for line in f:
            line = line.strip()
            if not line:
                break  # end of 1-grams
            parts = line.split('\t')
            unigram_probs.append(float(parts[0]))
            vocab.append(parts[1])

        while not f.readline().startswith('\\2-grams:'):
            continue
        bigrams = defaultdict(list)
        for line in f:
            line = line.strip()
            if not line:
                break  # end of 2-grams
            parts = line.split('\t')
            prob = float(parts[0])
            a, b = parts[1].split(' ')
            bigrams[a].append((prob, b))

        return vocab, np.array(unigram_probs) * LOG10, bigrams


def encode_bigrams(bigrams, model):
    encoded_bigrams = {}
    for prev, nexts in bigrams.items():
        prev_id = model.vocab_index(prev)
        next_ids = []
        for prob, b in nexts:
            next_id = model.vocab_index(b)
            next_ids.append((prob, next_id))
        encoded_bigrams[prev_id] = next_ids
    def pull_2nd(lst):
        return [x[1] for x in lst]
    unfiltered_bigrams = {a: pull_2nd(nexts) for a, nexts in encoded_bigrams.items()}
    # Most common bigrams (sorted by probability)
    filtered_bigrams = {a: pull_2nd(heapq.nlargest(100, nexts)) for a, nexts in encoded_bigrams.items()}
    return unfiltered_bigrams, filtered_bigrams



class Model:
    preloaded = {}
    @classmethod
    def preload_model(cls, name, basename):
        cls.preloaded[name] = cls.from_basename(name, basename)

    @classmethod
    def get_model(cls, name: str) -> 'Model':
        try:
            return cls.preloaded[name]
        except IndexError:
            raise Exception(f"The requested model `{name}` was not preloaded.")

    @classmethod
    def get_or_load_model(cls, name: str) -> 'Model':
        from suggestion.paths import paths
        if name not in cls.preloaded:
            cls.preload_model(name, paths.model_basename(name))
        return cls.get_model(name)


    @classmethod
    def from_basename(cls, name, basename):
        return cls(name=name, model_file=str(basename) + '.kenlm', arpa_file=str(basename) + '.arpa')


    def __init__(self, name, model_file, arpa_file):
        self.name = name
        self.model_file = model_file
        self.arpa_file = arpa_file
        self._load()

    def __reduce__(self):
        return Model.get_model, (self.name,)

    def _load(self):
        print("Loading model", self.name, '...', file=sys.stderr, end='')
        self.model = kenlm.LanguageModel(self.model_file)

        print(" reading raw ARPA data ... ", file=sys.stderr, end='')
        self.id2str, self.unigram_probs, bigrams = get_arpa_data(self.arpa_file)
        self.is_special = np.zeros(len(self.id2str), dtype=bool)
        for i, word in enumerate(self.id2str):
            assert self.model.vocab_index(word) == i, i
            if word[0] not in string.ascii_lowercase:
                self.is_special[i] = True
        # Since we give rare-word bonuses, count special words as super-common.
        self.unigram_probs_wordsonly = self.unigram_probs.copy()
        self.unigram_probs_wordsonly[self.is_special] = 0
        # ... but for finding the most common fallback words, count special words as impossible.
        unigram_probs_wordsonly_2 = self.unigram_probs.copy()
        unigram_probs_wordsonly_2[self.is_special] = -np.inf
        self.most_common_words_by_idx = np.argsort(unigram_probs_wordsonly_2)[-500:]
        print(" Encoding bigrams to indices... ", file=sys.stderr, end='')
        self.unfiltered_bigrams, self.filtered_bigrams = encode_bigrams(bigrams, self.model)

        # Vocab trie
        self.vocab_trie = datrie.BaseTrie(set(itertools.chain.from_iterable(self.id2str)))
        for i, s in enumerate(self.id2str):
            self.vocab_trie[s] = i

        self.eos_idx = self.model.vocab_index('</S>')
        self.eop_idx = self.model.vocab_index('</s>')
        print("Loaded.", file=sys.stderr)

    def prune_bigrams(self):
        # Filter bigrams to only include words that actually follow
        bigrams = self.unfiltered_bigrams
        while True:
            new_bigrams = {k: [tok for tok in v if len(bigrams.get(tok, [])) > 0] for k, v in bigrams.items()}
            new_bigrams_trim = {k: v for k, v in new_bigrams.items() if len(v) > 0}
            if len(new_bigrams) == len(new_bigrams_trim):
                break
            bigrams = new_bigrams_trim
        self.unfiltered_bigrams = bigrams

    def _compute_pos(self):
        print("Computing pos tags")
        pos_tags = [nltk.pos_tag([w or "UNK"], tagset='universal')[0][1] for w in self.id2str]
        self._id2tag = sorted(set(pos_tags))
        tag2id = {tag: id for id, tag in enumerate(self._id2tag)}
        self._pos_tags = np.array([tag2id[tag] for tag in pos_tags])

    @property
    def pos_tags(self):
        if not hasattr(self, '_pos_tags'):
            self._compute_pos()
        return self._pos_tags

    @property
    def id2tag(self):
        if not hasattr(self, '_id2tag'):
            self._compute_pos()
        return self._id2tag

    @property
    def word_lengths(self):
        if not hasattr(self, '_word_lengths'):
            self._word_lengths = np.array([len(w) if w is not None else 0 for w in self.id2str])
        return self._word_lengths

    @property
    def bos_state(self):
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)
        return state

    @property
    def null_context_state(self):
        state = kenlm.State()
        self.model.NullContextWrite(state)
        return state

    def get_state(self, words, bos=False):
        if bos:
            state = self.bos_state
        else:
            state = self.null_context_state
        score, state = self.score_seq(state, words)
        return state, score

    def score_seq(self, state, words):
        score = 0.
        for word in words:
            new_state = kenlm.State()
            score += self.model.base_score_from_idx(state, self.model.vocab_index(word), new_state)
            state = new_state
        return score * LOG10, state

    def score_seq_by_word(self, state, words):
        scores = []
        for word in words:
            new_state = kenlm.State()
            scores.append(LOG10 * self.model.base_score_from_idx(state, self.model.vocab_index(word), new_state))
            state = new_state
        return scores

    def advance_state(self, state, tok):
        new_state = kenlm.State()
        return new_state, LOG10 * self.model.base_score_from_idx(state, self.model.vocab_index(tok), new_state)

    def next_word_logprobs_raw(self, state, prev_word, prefix_logprobs=None):
        bigrams = self.unfiltered_bigrams
        if prefix_logprobs is not None:
            next_words = []
            prior_logprobs = []
            for logprob, prefix in prefix_logprobs:
                for word, word_idx in self.vocab_trie.items(prefix):
                    next_words.append(word_idx)
                    prior_logprobs.append(logprob)
        else:
            next_words = bigrams.get(self.model.vocab_index(prev_word), [])
            if len(next_words) == 0:
                next_words = bigrams.get(self.model.vocab_index('<S>'), [])
            next_words = [w for w in next_words if w != self.eos_idx and w != self.eop_idx]
        if len(next_words) == 0:
            return [], np.zeros(0)
        logprobs = self.eval_logprobs_for_words(state, next_words)
        if prefix_logprobs is not None:
            logprobs += prior_logprobs
        return next_words, logprobs

    def eval_logprobs_for_words(self, state, next_words):
        new_state = kenlm.State()
        logprobs = np.empty(len(next_words))
        for next_idx, word_idx in enumerate(next_words):
            logprobs[next_idx] = self.model.base_score_from_idx(state, word_idx, new_state)
        logprobs *= LOG10
        return logprobs


DEFAULT_SENTIMENT_WEIGHTS = [-1, -1, 0, 1, 1.]


class LMClassifier:
    def __init__(self, models, prior_counts, sentiment_weights=DEFAULT_SENTIMENT_WEIGHTS):
        self.models = models
        self.prior_logprobs = np.log(prior_counts / prior_counts.sum())
        sentiment_weights = np.array(sentiment_weights)
        sentiment_weights -= np.min(sentiment_weights)
        sentiment_weights /= np.max(sentiment_weights)
        self.sentiment_weights = sentiment_weights

    def get_state(self, toks, bos=False):
        models = self.models
        return [model.get_state(toks, bos=bos)[0] for model in models], np.zeros(len(models))

    def advance_state(self, state, tok):
        lm_states, scores = state
        new_lm_states = []
        score_deltas = np.empty(len(lm_states))
        for i, (lm_state, model) in enumerate(zip(lm_states, self.models)):
            new_lm_state, score_delta = model.advance_state(lm_state, tok)
            new_lm_states.append(new_lm_state)
            score_deltas[i] = score_delta
        new_state = new_lm_states, scores + score_deltas
        return new_state, score_deltas

    def classify_seq(self, state, toks):
        logprobs = self.prior_logprobs.copy()
        for tok in toks:
            state, score_deltas = self.advance_state(state, tok)
            logprobs += score_deltas
        logprobs -= logsumexp(logprobs)
        return np.exp(logprobs)

    def get_cur_posterior(self, state):
        lm_states, scores = state
        logprobs = scores + self.prior_logprobs
        probs = np.exp(logprobs)
        probs /= probs.sum()
        return probs

    def get_cur_classification(self, state):
        return self.get_cur_posterior(state) @ self.sentiment_weights

    def classify_seq_by_tok(self, state, toks):
        logprobs = self.prior_logprobs.copy()
        all_logprobs = []
        for tok in toks:
            state, score_deltas = self.advance_state(state, tok)
            logprobs = logprobs + score_deltas
            all_logprobs.append(logprobs)
        all_logprobs = np.array(all_logprobs)
        all_logprobs -= logsumexp(all_logprobs, axis=1, keepdims=True)
        return np.exp(all_logprobs)

    def sentiment(self, state, toks):
        probs = self.classify_seq(state, toks)
        return probs @ self.sentiment_weights

    def tok_weighted_sentiment(self, state, toks):
        return np.mean(self.classify_seq_by_tok(state, toks) @ self.sentiment_weights)
