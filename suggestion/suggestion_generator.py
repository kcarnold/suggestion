import kenlm
import heapq
import pickle
import os
import sys
import itertools
import datrie
import string
from collections import defaultdict
import numpy as np
import nltk
import cytoolz
import joblib

from .paths import paths
from .tokenization import tokenize_mid_document
from . import suffix_array, clustering

LOG10 = np.log(10)


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
        self.unigram_probs_wordsonly = self.unigram_probs.copy()
        for i, word in enumerate(self.id2str):
            assert self.model.vocab_index(word) == i, i
            if word[0] not in string.ascii_lowercase:
                self.unigram_probs_wordsonly[i] = 0
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
        return cls(model_file=basename + '.kenlm', arpa_file=basename + '.arpa')

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


models = {name: Model.from_basename(paths.model_basename(name)) for name in ['yelp_train']}
def get_model(name):
    return models[name]


if True:
    print("Loading docs...", end='', file=sys.stderr, flush=True)
    docs = pickle.load(open(os.path.join(paths.models, 'tokenized_reviews.pkl'), 'rb'))
    print(', suffix array...', end='', file=sys.stderr, flush=True)
    sufarr = suffix_array.DocSuffixArray(docs=docs, **joblib.load(os.path.join(paths.models, 'yelp_train_sufarr.joblib')))
    docs_by_id_fname = os.path.join(paths.models, 'yelp_train_docs_by_id.pkl')
    if os.path.exists(docs_by_id_fname):
        print(', loading id-mapped docs...', end='', file=sys.stderr, flush=True)
        docs_by_id = pickle.load(open(docs_by_id_fname, 'rb'))
    else:
        print(', mapping ids...', end='', file=sys.stderr, flush=True)
        _str2id = {word: idx for idx, word in enumerate(models['yelp_train'].id2str)}
        docs_by_id = [[_str2id.get(word, 0) for word in doc] for doc in docs]
        pickle.dump(docs_by_id, open(docs_by_id_fname, 'wb'), -1)
    print(" Done.", file=sys.stderr)

if True:
    print("Loading clusterizer...", end='', file=sys.stderr, flush=True)
    clizer_data = joblib.load('clizer_core.joblib', mmap_mode='r')
    clizer = clustering.Clusterizer(**clizer_data,
        sents=pickle.load(open('clizer_sents.pkl', 'rb')),
        vectorizer=pickle.load(open('clizer_vectorizer.pkl', 'rb')),
        unique_starts=pickle.load(open('clizer_unique_starts.pkl', 'rb')))
    print("Done.", file=sys.stderr)


import numba
@numba.jit
def _next_elt_le(arr, criterion, start, end):
    for i in range(start, end):
        if arr[i] <= criterion:
            return i
    return end

def collect_words_in_range(start, after_end, word_idx, docs):
    words = []
    if start == after_end:
        return words
    word = docs[sufarr.doc_idx[start]][sufarr.tok_idx[start] + word_idx]
    words.append(word)
    while True:
        before_next_idx = _next_elt_le(sufarr.lcp, word_idx, start, after_end - 1)
        if before_next_idx == after_end - 1:
            break
        next_idx = before_next_idx + 1
        word = docs[sufarr.doc_idx[next_idx]][sufarr.tok_idx[next_idx] + word_idx]
        words.append(word)
        start = next_idx
    return words



from scipy.misc import logsumexp
def softmax(scores):
    return np.exp(scores - logsumexp(scores))


def next_word_probs(model, state, prev_word, prefix_logprobs=None, temperature=1., length_bonus_min_length=6, length_bonus_amt=0., pos_weights=None):
    next_words, logprobs = model.next_word_logprobs_raw(state, prev_word, prefix_logprobs=prefix_logprobs)
    if len(next_words) == 0:
        return next_words, logprobs
    if length_bonus_amt:
        length_bonus_elegible = model.word_lengths[next_words] >= length_bonus_min_length
        logprobs = logprobs + length_bonus_amt * length_bonus_elegible
    if pos_weights is not None:
        poses = model.pos_tags[next_words]
        logprobs = logprobs + pos_weights[poses]
    logprobs /= temperature
    return next_words, softmax(logprobs)


class GenerationFailedException(Exception):
    pass

def retry_on_exception(exception, tries):
    def decorator(fn):
        def wrapper(*a, **kw):
            for i in range(tries):
                try:
                    return fn(*a, **kw)
                except exception:
                    continue
                except:
                    raise
            return fn(*a, **kw)
        return wrapper
    return decorator

@retry_on_exception(GenerationFailedException, 10)
def generate_phrase(model, context_toks, length, prefix_logprobs=None, **kw):
    if context_toks[0] == '<s>':
        state, _ = model.get_state(context_toks[1:], bos=True)
    else:
        state, _ = model.get_state(context_toks, bos=False)
    phrase = context_toks[:]
    generated_logprobs = np.empty(length)
    for i in range(length):
        next_words, probs = next_word_probs(model, state, phrase[-1], prefix_logprobs=prefix_logprobs, **kw)
        if len(next_words) == 0:
            raise GenerationFailedException
        prefix_logprobs = None
        picked_subidx = np.random.choice(len(probs), p=probs)
        picked_idx = next_words[picked_subidx]
        new_state = kenlm.State()
        model.model.base_score_from_idx(state, picked_idx, new_state)
        state = new_state
        word = model.id2str[picked_idx]
        phrase.append(word)
        generated_logprobs[i] = np.log(probs[picked_subidx])
    return phrase[len(context_toks):], generated_logprobs


def generate_phrase_from_sufarr(model, sufarr, context_toks, length, prefix_logprobs=None, temperature=1.):
    if context_toks[0] == '<s>':
        state, _ = model.get_state(context_toks[1:], bos=True)
    else:
        state, _ = model.get_state(context_toks, bos=False)
    phrase = []
    generated_logprobs = np.empty(length)
    for i in range(length):
        start_idx, end_idx = sufarr.search_range((context_toks[-1],) + tuple(phrase) + ('',))
        next_words = collect_words_in_range(start_idx, end_idx, i + 1)

        if prefix_logprobs is not None:
            prior_logprobs = np.full(len(next_words), -10)
            for logprob, prefix in prefix_logprobs:
                for nextword_idx, word in enumerate(next_words):
                    if word.startswith(prefix):
                        prior_logprobs[nextword_idx] = logprob
        else:
            prior_logprobs = None
        if len(next_words) == 0:
            raise GenerationFailedException
        vocab_indices = [model.model.vocab_index(word) for word in next_words]
        logprobs = model.eval_logprobs_for_words(state, vocab_indices)
        if prior_logprobs is not None:
            logprobs += prior_logprobs
        logprobs /= temperature
        probs = softmax(logprobs)

        picked_subidx = np.random.choice(len(probs), p=probs)
        picked_idx = vocab_indices[picked_subidx]
        new_state = kenlm.State()
        model.model.base_score_from_idx(state, picked_idx, new_state)
        state = new_state
        word = next_words[picked_subidx]
        phrase.append(word)
        generated_logprobs[i] = np.log(probs[picked_subidx])
        prefix_logprobs = None
    return phrase, generated_logprobs



def generate_diverse_phrases(model, context_toks, n, length, prefix_logprobs=None, **kw):
    if model is None:
        model = 'yelp_train'
    if isinstance(model, str):
        model = get_model(model)
    if 'pos_weights' in kw:
        kw['pos_weights'] = np.array(kw['pos_weights'])

    state, _ = model.get_state(context_toks)
    first_words, first_word_probs = next_word_probs(model, state, context_toks[-1], prefix_logprobs=prefix_logprobs, **kw)
    if len(first_words) == 0:
        return []
    res = []
    for idx in np.random.choice(len(first_words), min(len(first_words), n), p=first_word_probs, replace=False):
        first_word = model.id2str[first_words[idx]]
        first_word_logprob = np.log(first_word_probs[idx])
#        phrase, phrase_logprobs = generate_phrase(model, context_toks + [first_word], length - 1, **kw)
        phrase, phrase_logprobs = generate_phrase_from_sufarr(model, sufarr, context_toks + [first_word], length - 1, **kw)
        res.append(([first_word] + phrase, np.hstack(([first_word_logprob], phrase_logprobs))))
    return res


from collections import namedtuple
BeamEntry = namedtuple("BeamEntry", 'score, words, done, penultimate_state, last_word_idx, num_chars, bonuses')

def beam_search_phrases(model, start_words, beam_width, length, prefix_logprobs=None, rare_word_bonus=0.):
    if isinstance(model, str):
        model = get_model(model)
    unigram_probs = model.unigram_probs_wordsonly
    start_state, start_score = model.get_state(start_words, bos=False)
    beam = [(0., [], False, start_state, model.model.vocab_index(start_words[-1]), 0, None)]
    for i in range(length):
        bigrams = model.unfiltered_bigrams if i == 0 else model.filtered_bigrams
        prefix_chars = 1 if i > 0 else 0
        DONE = 2
        new_beam = [ent for ent in beam if ent[DONE]]
        for entry in beam:
            score, words, done, penultimate_state, last_word_idx, num_chars, bonuses = entry
            if done:
                continue
            else:
                if i > 0:
                    last_state = kenlm.State()
                    model.model.base_score_from_idx(penultimate_state, last_word_idx, last_state)
                else:
                    last_state = penultimate_state
                probs = None
                if i == 0 and prefix_logprobs is not None:
                    next_words = []
                    probs = []
                    for prob, prefix in prefix_logprobs:
                        for word, word_idx in model.vocab_trie.items(prefix):
                            next_words.append(word_idx)
                            probs.append(prob)
                else:
                    # print(id2str[last_word])
                    next_words = bigrams.get(last_word_idx, [])
                new_state = kenlm.State()
                for next_idx, word_idx in enumerate(next_words):
                    if word_idx == model.eos_idx or word_idx == model.eop_idx:
                        continue
                    if probs is not None:
                        prob = probs[next_idx]
                    else:
                        prob = 0.
                    word = model.id2str[word_idx]
                    unigram_bonus = -unigram_probs[word_idx]*rare_word_bonus if i > 0 and word not in words else 0.
                    new_score = score + prob + unigram_bonus + LOG10 * model.model.base_score_from_idx(last_state, word_idx, new_state)
                    new_words = words + [word]
                    new_num_chars = num_chars + prefix_chars + len(word)
                    new_entry = (new_score, new_words, new_num_chars >= length, last_state, word_idx, new_num_chars, None)
                    if len(new_beam) == beam_width:
                        heapq.heapreplace(new_beam, new_entry)
                    else:
                        new_beam.append(new_entry)
                        if len(new_beam) == beam_width:
                            heapq.heapify(new_beam)
                    assert len(new_beam) <= beam_width
        beam = new_beam
    return [BeamEntry(*ent) for ent in sorted(beam, reverse=True)]


def beam_search_sufarr_init(model, start_words):
    start_state, start_score = model.get_state(start_words, bos=False)
    return [(0., [], False, start_state, None, 0, (0, len(sufarr.doc_idx)))]


def beam_search_sufarr_extend(model, beam, context_tuple, iteration_num, beam_width, target_length, rare_word_bonus=0., prefix=''):
    if isinstance(model, str):
        model = get_model(model)
    unigram_probs = model.unigram_probs_wordsonly
    def candidates():
        for entry in beam:
            score, words, done, penultimate_state, last_word_idx, num_chars, (prev_start_idx, prev_end_idx) = entry
            if done:
                yield entry
                continue
            if last_word_idx is not None:
                last_state = kenlm.State()
                model.model.base_score_from_idx(penultimate_state, last_word_idx, last_state)
            else:
                last_state = penultimate_state
            start_idx, end_idx = sufarr.search_range(context_tuple + tuple(words) + (prefix,), lo=prev_start_idx, hi=prev_end_idx)
            next_word_ids = collect_words_in_range(start_idx, end_idx, iteration_num + 1, docs_by_id)
            if len(next_word_ids) == 0:
                assert model.id2str[last_word_idx] == '</S>', "We only expect to run out of words at an end-of-sentence that's also an end-of-document."
                continue
            bound_indices = start_idx, end_idx
            new_state = kenlm.State()
            for next_idx, word_idx in enumerate(next_word_ids):
                if word_idx == 0: continue
                word = model.id2str[word_idx]
                new_words = words + [word]
                new_num_chars = num_chars + 1 + len(word)
                logprob = LOG10 * model.model.base_score_from_idx(last_state, word_idx, new_state)
                unigram_bonus = -unigram_probs[word_idx]*rare_word_bonus if word not in words else 0.
                new_score = score + logprob + unigram_bonus
                done = new_num_chars >= target_length
                yield new_score, new_words, done, last_state, word_idx, new_num_chars, bound_indices#bonuses + [unigram_bonus])
    return heapq.nlargest(beam_width, candidates())



def tap_decoder(char_model, before_cursor, cur_word, key_rects, beam_width=100, scale=100.):
    keys = [k['key'] for k in key_rects]
    rects = [k['rect'] for k in key_rects]
    centers = [((rect['left'] + rect['right']) / 2, (rect['top'] + rect['bottom']) / 2) for rect in rects]

    beam_width = 100
    beam = [(0., '', None)]
    for item in cur_word:
        if 'tap' not in item:
            letter = item['letter']
            letters_and_distances = [(letter, 0)]
        else:
            x, y = item['tap']
            sq_dist_to_center = [(x - rect_x) ** 2. + (y - rect_y) ** 2. for rect_x, rect_y in centers]
            letters_and_distances = zip(keys, sq_dist_to_center)
        new_beam = []
        # print(np.min(sq_dist_to_center) / scale, keys[np.argmin(sq_dist_to_center)])
        for score, sofar, penultimate_state in beam:
            last_state = kenlm.State()
            if sofar:
                char_model.BaseScore(penultimate_state, sofar[-1], last_state)
            else:
                char_model.NullContextWrite(last_state)
                for c in before_cursor:
                    next_state = kenlm.State()
                    char_model.BaseScore(last_state, c, next_state)
                    last_state = next_state
            next_state = kenlm.State()
            for key, dist in letters_and_distances:
                new_so_far = sofar + key
                new_beam.append((score + char_model.BaseScore(last_state, key, next_state) - dist / scale, new_so_far, last_state))
        beam = sorted(new_beam, reverse=True)[:beam_width]
    return [(prob, word) for prob, word, state in sorted(beam, reverse=True)[:10]]


def tokenize_sofar(sofar):
    toks = tokenize_mid_document(sofar.lower().replace(' .', '.').replace(' ,', ','))[0]
    if toks[-1] != '':
        print("WEIRD: somehow we got a mid-word sofar:", repr(sofar))
    assert toks[0] == "<D>"
    assert toks[1] == "<P>"
    assert toks[2] == "<S>"
    return ['<s>', "<D>"] + toks[3:-1]


def phrases_to_suggs(phrases):
    def de_numpy(x):
        return x.tolist() if x is not None else None
    return [dict(one_word=dict(words=phrase[:1]), continuation=[dict(words=phrase[1:])], probs=de_numpy(probs)) for phrase, probs in phrases]


def predict_forward(domain, toks, first_word, beam_width=50, length=30):
    model = get_model(domain)
    continuations = beam_search_phrases(model, toks + [first_word],
        beam_width=beam_width, length=length - len(first_word) - 1)
    if len(continuations) > 0:
        continuation = continuations[0].words
    else:
        continuation = []
    return [first_word] + continuation, None



def get_suggestions_async(executor, *, sofar, cur_word, domain, rare_word_bonus, use_sufarr, temperature, length=30, sug_state=None, **kw):
    model = get_model(domain)
    toks = tokenize_sofar(sofar)
    prefix_logprobs = [(0., ''.join(item['letter'] for item in cur_word))] if len(cur_word) > 0 else None
    prefix = ''.join(item['letter'] for item in cur_word)
    # prefix_probs = tap_decoder(sofar[-12:].replace(' ', '_'), cur_word, key_rects)
    use_bos_suggs = True
    if use_bos_suggs and len(cur_word) == 0 and toks[-1] in ['<D>', '<S>']:
        if sug_state is None:
            sug_state = {}
        if 'suggested_already' not in sug_state:
            sug_state['suggested_already'] = []
        suggested_already = sug_state['suggested_already']
        print("Already suggested", suggested_already)

        scores_by_cluster = clizer.scores_by_cluster.copy()
        scores_by_cluster[suggested_already] = -np.inf
        most_distinctive = np.argmax(scores_by_cluster, axis=0)

        def normal_lik(x, sigma):
            return np.exp(-.5*(x/sigma)**2) / (2*np.pi*sigma)

        sents = nltk.sent_tokenize(sofar)
        how_much_covered = np.zeros(clizer.n_clusters)

        for sent in sents:
            cluster_distrib = cytoolz.thread_first(
                [sent],
                clizer.vectorize_sents,
                clustering.normalize_vecs,
                clizer.clusterer.transform,
                (normal_lik, .5),
                clustering.normalize_dists
                )[0]
            how_much_covered += cluster_distrib

        topics_to_suggest = np.argsort(how_much_covered)[:3]
        print("Suggesting topics", topics_to_suggest.tolist())
        phrases = []
        for cluster_idx in topics_to_suggest:
            suggest_idx = most_distinctive[cluster_idx]
            suggested_already.append(suggest_idx)
            phrases.append((clizer.unique_starts[suggest_idx], None))
        return phrases, sug_state

    if temperature == 0:
        if use_sufarr:
            beam_width = 50
            beam = beam_search_sufarr_init(model, toks)
            context_tuple = (toks[-1],)
            for i in range(length):
                beam_chunks = cytoolz.partition_all(8, beam)
                parallel_futures = yield [executor.submit(
                    beam_search_sufarr_extend, domain, chunk, context_tuple, i, 25, length, rare_word_bonus=rare_word_bonus, prefix=prefix)
                    for chunk in beam_chunks]
                parallel_beam = list(cytoolz.concat(parallel_futures))
                prefix = ''
                beam = heapq.nlargest(beam_width, parallel_beam)
                # entry 2 is "DONE"
                if all(ent[2] for ent in beam):
                    break
            ents = [BeamEntry(*ent) for ent in beam]
            result = [ents.pop(0)]
            first_words = {ent.words[0] for ent in result}
            while len(result) < 3 and len(ents) > 0:
                ents.sort(reverse=True, key=lambda ent: (ent.words[0] not in first_words, ent.score))
                best = ents.pop(0)
                first_words.add(best.words[0])
                result.append(best)
            phrases = [([word for word in ent.words if word[0] != '<'], None) for ent in result]
        else:
            first_word_ents = yield executor.submit(beam_search_phrases, domain, toks, beam_width=100, length=1, prefix_logprobs=prefix_logprobs)
            next_words = [ent.words[0] for ent in first_word_ents[:3]]
            phrases = (yield [executor.submit(predict_forward, domain, toks, oneword_suggestion) for oneword_suggestion in next_words])
    else:
        # TODO: upgrade to use_sufarr flag
        phrases = generate_diverse_phrases(
            domain, toks, 3, 6, prefix_logprobs=prefix_logprobs, temperature=temperature, use_sufarr=use_sufarr, **kw)
    return phrases, sug_state


