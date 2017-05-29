from collections import defaultdict
import numpy as np
import heapq
import sys
import string
import datrie
import nltk
import itertools
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
    def __init__(self, model_file, arpa_file):
        self.model_file = model_file
        self.arpa_file = arpa_file
        self._load()

    def __getstate__(self):
        return dict(model_file=self.model_file, arpa_file=self.arpa_file)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load()

    def _load(self):
        print("Loading model", file=sys.stderr)
        self.model = kenlm.LanguageModel(self.model_file)
        print("...done.", file=sys.stderr)

        print("Reading raw ARPA data", file=sys.stderr)
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
        print("Encoding bigrams to indices", file=sys.stderr)
        self.unfiltered_bigrams, self.filtered_bigrams = encode_bigrams(bigrams, self.model)

        # Vocab trie
        self.vocab_trie = datrie.BaseTrie(set(itertools.chain.from_iterable(self.id2str)))
        for i, s in enumerate(self.id2str):
            self.vocab_trie[s] = i

        self.eos_idx = self.model.vocab_index('</S>')
        self.eop_idx = self.model.vocab_index('</s>')

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


    @classmethod
    def from_basename(cls, basename):
        return cls(model_file=str(basename) + '.kenlm', arpa_file=str(basename) + '.arpa')

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
