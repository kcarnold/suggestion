import kenlm
import heapq
import pickle
import os
import sys
import numpy as np
import nltk
import cytoolz
import joblib
from scipy.misc import logsumexp

from .paths import paths
from .tokenization import tokenize_mid_document
from .lang_model import Model, LMClassifier
from .diversity import scalar_diversity
from . import suffix_array, clustering, manual_bos

LOG10 = np.log(10)

'''
yelp_train-3star
tweeterinchief
'''

PRELOAD_MODELS = '''
yelp_train
yelp_train-1star
yelp_train-2star
yelp_train-4star
yelp_train-5star
yelp_topic_seqs
sotu'''.split()
[Model.preload_model(name, paths.model_basename(name)) for name in PRELOAD_MODELS]
get_model = Model.get_model

CLASSIFIERS = {'positive': LMClassifier([
    get_model(f'yelp_train-{star}star') for star in [1, 2, 4, 5]], [-.5, -.5, .5, .5])}


enable_sufarr = False
enable_bos_suggs = False

if enable_sufarr:
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
        _str2id = {word: idx for idx, word in enumerate(get_model('yelp_train').id2str)}
        docs_by_id = [[_str2id.get(word, 0) for word in doc] for doc in docs]
        pickle.dump(docs_by_id, open(docs_by_id_fname, 'wb'), -1)
    print(" Done.", file=sys.stderr)

if enable_bos_suggs:
    print("Loading goal-oriented suggestion data...", end='', file=sys.stderr, flush=True)
    with open(os.path.join(paths.parent, 'models', 'goal_oriented_suggestion_data.pkl'), 'rb') as f:
        clizer = pickle.load(f)
    clizer.topic_continuation_scores = np.load('topic_continuation_scores.npy')

    keep = np.full(len(clizer.unique_starts), True, dtype=bool)
    keep[clizer.omit] = 0
    np.mean(keep)
    clizer.scores_by_cluster = clizer.scores_by_cluster[keep]
    clizer.topic_continuation_scores = clizer.topic_continuation_scores[keep]
    clizer.unique_starts = [clizer.unique_starts[i] for i in np.flatnonzero(keep)]

    likelihood_bias = logsumexp(clizer.scores_by_cluster, axis=1, keepdims=True)
    clizer.scores_by_cluster = clizer.scores_by_cluster - .85 * likelihood_bias

    clizer.topic_continuation_scores_argsort = [np.argsort(clizer.topic_continuation_scores[:,topic])[::-1] for topic in range(clizer.n_clusters)]
    clizer.scores_by_cluster_argsort = [np.argsort(clizer.scores_by_cluster[:,topic])[::-1] for topic in range(clizer.n_clusters)]

    del keep
    del likelihood_bias


    topic_tags = [f'<T{i}>' for i in range(clizer.n_clusters)]
    topic_seq_model = get_model('yelp_topic_seqs')
    topic_word_indices = [topic_seq_model.model.vocab_index(tag) for tag in topic_tags]
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
BeamEntry = namedtuple("BeamEntry", 'score, words, done, penultimate_state, last_word_idx, num_chars, extra')

def beam_search_phrases_init(model, start_words, *, classifiers=[], **kw):
    if isinstance(model, str):
        model = get_model(model)
    classifiers = [(CLASSIFIERS[clf], weight) if isinstance(clf, str) else (clf, weight) for clf, weight in classifiers]
    start_state, start_score = model.get_state(start_words, bos=False)
    classifier_states = [clf.get_state(start_words, bos=False) for clf, weight in classifiers]
    return [(0., [], False, start_state, model.model.vocab_index(start_words[-1]), 0, classifier_states)]


def beam_search_phrases_extend(model, beam, *, beam_width, iteration_num, length_after_first, prefix_logprobs=None, rare_word_bonus=0., constraints, classifiers=[]):
    if isinstance(model, str):
        model = get_model(model)
    classifiers = [(CLASSIFIERS[clf], weight) if isinstance(clf, str) else (clf, weight) for clf, weight in classifiers]
    unigram_probs = model.unigram_probs_wordsonly
    avoid_letter = constraints.get('avoidLetter')

    bigrams = model.unfiltered_bigrams if iteration_num == 0 else model.filtered_bigrams
    DONE = 2
    new_beam = [ent for ent in beam if ent[DONE]]
    for entry in beam:
        score, words, done, penultimate_state, last_word_idx, num_chars, classifier_states = entry
        if done:
            continue
        else:
            if iteration_num > 0:
                last_state = kenlm.State()
                model.model.base_score_from_idx(penultimate_state, last_word_idx, last_state)
                # penultimate_classifier_states = [clf.advance_state(clf_state, words[-1])
                #     for (clf, weight), clf_state in zip(classifiers, classifier_states)]
            else:
                last_state = penultimate_state
                # penultimate_classifier_states = classifier_states
            probs = None
            if iteration_num == 0 and prefix_logprobs is not None:
                next_words = []
                probs = []
                for prob, prefix in prefix_logprobs:
                    for word, word_idx in model.vocab_trie.items(prefix):
                        next_words.append(word_idx)
                        probs.append(prob)
            else:
                # print(id2str[last_word])
                next_words = bigrams.get(last_word_idx, [])
                if len(next_words) < 10:
                    if iteration_num == 0:
                        # Fall back to all common words.
                        next_words = model.most_common_words_by_idx
                    else:
                        # Use the larger set of possible next words
                        next_words = model.unfiltered_bigrams.get(last_word_idx, [])
                        if len(next_words) < 10:
                            next_words = model.most_common_words_by_idx
            new_state = kenlm.State()
            for next_idx, word_idx in enumerate(next_words):
                if word_idx == model.eos_idx or word_idx == model.eop_idx:
                    continue
                if probs is not None:
                    prob = probs[next_idx]
                else:
                    prob = 0.
                word = model.id2str[word_idx]
                if avoid_letter is not None and avoid_letter in word:
                    continue
                if word[0] in '.?!':
                    continue
                unigram_bonus = -unigram_probs[word_idx]*rare_word_bonus if iteration_num > 0 and word not in words else 0.
                main_model_score = LOG10 * model.model.base_score_from_idx(last_state, word_idx, new_state)
                new_classifier_states = []
                classifier_score = 0.
                for (clf, weight), clf_state in zip(classifiers, classifier_states):
                    clf_state, score_delta = clf.advance_state(clf_state, word)
                    new_classifier_states.append(clf_state)
                    classifier_score += weight * score_delta
                new_score = score + prob + unigram_bonus + main_model_score + classifier_score
                new_words = words + [word]
                new_num_chars = num_chars + 1 + len(word) if iteration_num else 0
                done = new_num_chars >= length_after_first
                new_entry = (new_score, new_words, done, last_state, word_idx, new_num_chars, classifier_states)
                if len(new_beam) == beam_width:
                    heapq.heappushpop(new_beam, new_entry)
                else:
                    new_beam.append(new_entry)
                    if len(new_beam) == beam_width:
                        heapq.heapify(new_beam)
                assert len(new_beam) <= beam_width
    return new_beam


def beam_search_phrases_loop(model, beam, *, length_after_first, prefix_logprobs=None, start_idx=0, **kw):
    for iteration_num in range(start_idx, length_after_first):
        beam = beam_search_phrases_extend(model, beam, iteration_num=iteration_num, length_after_first=length_after_first,
            prefix_logprobs=prefix_logprobs, **kw)
        prefix_logprobs = None
    return [BeamEntry(*ent) for ent in sorted(beam, reverse=True)]


def beam_search_phrases(model, start_words, **kw):
    beam = beam_search_phrases_init(model, start_words, **kw)
    beam = beam_search_phrases_loop(model, beam, **kw)
    beam.sort(reverse=True)
    return [BeamEntry(*ent) for ent in beam]


def beam_search_sufarr_init(model, start_words):
    start_state, start_score = model.get_state(start_words, bos=False)
    return [(0., [], False, start_state, None, 0, (0, len(sufarr.doc_idx), model.null_context_state))]


def beam_search_sufarr_extend(model, beam, context_tuple, iteration_num, beam_width, length_after_first, *, word_bonuses=None, prefix='', null_logprob_weight=0., constraints):
    if isinstance(model, str):
        model = get_model(model)
    avoid_letter = constraints.get('avoidLetter')
    def candidates():
        for entry in beam:
            score, words, done, penultimate_state, last_word_idx, num_chars, (prev_start_idx, prev_end_idx, penultimate_state_null) = entry
            if done:
                yield entry
                continue
            if last_word_idx is not None:
                last_state = kenlm.State()
                model.model.base_score_from_idx(penultimate_state, last_word_idx, last_state)
                last_state_null = kenlm.State()
                model.model.base_score_from_idx(penultimate_state_null, last_word_idx, last_state_null)
            else:
                last_state = penultimate_state
                last_state_null = penultimate_state_null
            start_idx, end_idx = sufarr.search_range(context_tuple + tuple(words) + (prefix,), lo=prev_start_idx, hi=prev_end_idx)
            next_word_ids = collect_words_in_range(start_idx, end_idx, iteration_num + 1, docs_by_id)
            if len(next_word_ids) == 0:
                assert iteration_num == 0 or model.id2str[last_word_idx] == '</S>', "We only expect to run out of words at an end-of-sentence that's also an end-of-document."
                continue
            new_state = kenlm.State()
            for next_idx, word_idx in enumerate(next_word_ids):
                if word_idx == 0: continue
                word = model.id2str[word_idx]
                if avoid_letter is not None and avoid_letter in word:
                    continue
                if word[0] in '.?!':
                    continue
                new_words = words + [word]
                new_num_chars = num_chars + 1 + len(word) if iteration_num > 0 else 0
                logprob = LOG10 * model.model.base_score_from_idx(last_state, word_idx, new_state)
                unigram_bonus = word_bonuses[word_idx] if word not in words else 0.
                logprob_null = LOG10 * model.model.base_score_from_idx(last_state_null, word_idx, new_state)
                new_score = score + logprob + null_logprob_weight * logprob_null + unigram_bonus
                done = new_num_chars >= length_after_first
                yield new_score, new_words, done, last_state, word_idx, new_num_chars, (start_idx, end_idx, last_state_null)#bonuses + [unigram_bonus])
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
    return [dict(one_word=dict(words=phrase[:1]), continuation=[dict(words=phrase[1:])], meta=meta) for phrase, meta in phrases]


def predict_forward(domain, toks, beam_width, length_after_first, constraints, classifiers=[]):
    model = get_model(domain)
    first_word = toks[-1]
    if first_word in '.?!':
        return [first_word], None
    continuations = beam_search_phrases(model, toks,
        beam_width=beam_width, length_after_first=length_after_first, constraints=constraints, classifiers=classifiers)
    if len(continuations) > 0:
        continuation = continuations[0].words
    else:
        continuation = []
    return [first_word] + continuation, None



def try_to_match_topic_distribution(clizer, target_dist, sents):
    def normal_lik(x, sigma):
        return np.exp(-.5*(x/sigma)**2) / (2*np.pi*sigma)

    sent_cluster_distribs = cytoolz.thread_first(
        sents,
        clizer.vectorize_sents,
        clustering.normalize_vecs,
        clizer.clusterer.transform,
        (normal_lik, .5),
        clustering.normalize_dists
        )

    new_dists_opts = np.eye(clizer.n_clusters)

    from scipy.special import kl_div
    with_new_dist = np.array([np.concatenate((sent_cluster_distribs, new_dist_opt[None]), axis=0) for new_dist_opt in new_dists_opts])
    dist_with_new_dist = clustering.normalize_dists(np.mean(with_new_dist, axis=1))
    return np.argsort(kl_div(dist_with_new_dist, target_dist).sum(axis=1))[:3].tolist()

def get_topic_seq(sents):
    if len(sents) == 0:
        return []
    cluster_distances = cytoolz.thread_first(
        sents,
        clizer.vectorize_sents,
        clustering.normalize_vecs,
        clizer.clusterer.transform)
    return np.argmin(cluster_distances, axis=1).tolist()


def get_bos_suggs(sofar, sug_state, *, bos_sugg_flag, constraints, verbose=False):
    if sug_state is None:
        sug_state = {}
    if 'suggested_already' not in sug_state:
        sug_state['suggested_already'] = set()
    suggested_already = sug_state['suggested_already']
    if verbose:
        print("Already suggested", suggested_already)

    sents = nltk.sent_tokenize(sofar)

    topic_seq = get_topic_seq(sents)
    if False:
        topics_to_suggest = try_to_match_topic_distribution(
            clizer=clizer,
            target_dist=clizer.target_dists['best'],
            sents=sents)
    elif bos_sugg_flag == 'continue':
        if len(topic_seq) == 0:
            return None, sug_state, None
        last_topic = topic_seq[-1]
        topics_to_suggest = [last_topic] * 3
    else:
        # Find the most likely next topics.
        topic_seq_state = topic_seq_model.get_state([topic_tags[topic] for topic in topic_seq], bos=True)[0]
        topic_likelihood = topic_seq_model.eval_logprobs_for_words(topic_seq_state, topic_word_indices)
        if len(topic_seq):
            # Ensure that we don't continue the same topic.
            last_topic = topic_seq[-1]
            topic_likelihood[last_topic] = -np.inf

            # Penalize already-covered topics.
            for topic in topic_seq:
                topic_likelihood[topic] -= 100.0

        topics_to_suggest = np.argsort(topic_likelihood)[-3:][::-1].tolist()

    if verbose:
        print(f"seq={topic_seq} flag={bos_sugg_flag} suggesting={topics_to_suggest}")

    if bos_sugg_flag == 'continue':
        scores_by_cluster = clizer.topic_continuation_scores
        argsort_scores_for_topic = clizer.topic_continuation_scores_argsort
    else:
        scores_by_cluster = clizer.scores_by_cluster
        argsort_scores_for_topic = clizer.scores_by_cluster_argsort

    avoid_letter = constraints.get('avoidLetter')
    phrases = []
    first_words = []
    for topic in topics_to_suggest:
        # Try to find a start for this topic that doesn't overlap an existing one in first word.
        for suggest_idx in argsort_scores_for_topic[topic]:
            phrase = clizer.unique_starts[suggest_idx]
            if phrase[0] in first_words:
                continue
            beginning = ' '.join(phrase[:3])
            if beginning in suggested_already:
                continue
            if avoid_letter is not None and (avoid_letter in beginning or avoid_letter in ''.join(phrase)):
                continue
            first_words.append(phrase[0])
            suggested_already.add(beginning)
            phrases.append((phrase, 'bos'))
            break
    return phrases, sug_state, dict(topic_seq=topic_seq, topics_to_suggest=topics_to_suggest)


def get_sentence_enders(model, start_words):
    if isinstance(model, str):
        model = get_model(model)
    start_state, start_score = model.get_state(start_words, bos=False)
    toks = list('.?!')
    end_indices = [model.model.vocab_index(tok) for tok in toks]
    scores = np.exp(model.eval_logprobs_for_words(start_state, end_indices))
    cum_score = np.sum(scores)
    if cum_score > 2/3:
        return [toks[i] for i in np.argsort(scores)[::-1][:2]]
    elif cum_score > 1/3:
        return [toks[np.argmax(scores)]]
    return []


def get_suggestions_async(executor, *, sofar, cur_word, domain,
    rare_word_bonus, use_sufarr, temperature, use_bos_suggs,
    length_after_first=17, sug_state=None, word_bonuses=None, prewrite_info=None,
    constraints={},
    promise=None,
    polarity_split=None,
    **kw):

    model = get_model(domain)
    toks = tokenize_sofar(sofar)
    prefix_logprobs = [(0., ''.join(item['letter'] for item in cur_word))] if len(cur_word) > 0 else None
    prefix = ''.join(item['letter'] for item in cur_word)
    # prefix_probs = tap_decoder(sofar[-12:].replace(' ', '_'), cur_word, key_rects)

    if word_bonuses is None and prewrite_info is not None:
        known_words = set()
        unknown_words = set()
        word_bonuses = np.zeros(len(model.id2str))
        unigram_probs = model.unigram_probs_wordsonly
        for word in prewrite_info['text'].split():
            idx = model.model.vocab_index(word)
            if idx != 0:
                word_bonuses[idx] = prewrite_info['amount'] * -unigram_probs[idx]
                known_words.add(word)
            else:
                unknown_words.add(word)
        print(f"Bonusing {len(known_words)} prewrite words: {' '.join(sorted(known_words))}")
        print(f"Not bonusing {len(unknown_words)} unknown words: {' '.join(sorted(unknown_words))}")

    # Beginning of sentence suggestions
    if use_bos_suggs and use_bos_suggs != 'manual' and not enable_bos_suggs:
        print("Warning: requested BOS suggs but they're not enabled.")
        use_bos_suggs = False
    if use_bos_suggs and len(cur_word) == 0 and toks[-1] in ['<D>', '<S>']:
        if promise is not None:
            print("Warning: promise enabled but making beginning-of-sentence suggestions!")
        if use_bos_suggs == 'manual':
            phrases, sug_state = manual_bos.get_manual_bos(sofar, sug_state or {})
        else:
            phrases, sug_state, _ = get_bos_suggs(sofar, sug_state, bos_sugg_flag=use_bos_suggs, constraints=constraints)
        if phrases is not None:
            return phrases, sug_state

    if use_sufarr and not enable_sufarr:
        print("Warning: requested sufarr but not enabled.")
        use_sufarr = False

    if temperature == 0:
        if use_sufarr and len(cur_word) == 0:
            assert polarity_split is None, "sufarr doesn't support polarity_split yet"
            assert promise is None, "sufarr doesn't support promises yet"
            beam_width = 100
            beam = beam_search_sufarr_init(model, toks)
            context_tuple = (toks[-1],)
            if word_bonuses is None:
                # The multiplication makes a copy.
                word_bonuses = model.unigram_probs_wordsonly * -rare_word_bonus
            else:
                word_bonuses = word_bonuses.copy()
            # Don't double-bonus words that have already been used.
            for word in set(toks):
                word_idx = model.model.vocab_index(word)
                word_bonuses[word_idx] = 0.
            for i in range(length_after_first):
                beam_chunks = cytoolz.partition_all(8, beam)
                parallel_futures = yield [executor.submit(
                    beam_search_sufarr_extend, domain, chunk, context_tuple, i, beam_width, length_after_first=length_after_first, word_bonuses=word_bonuses, prefix=prefix, constraints=constraints, **kw)
                    for chunk in beam_chunks]
                parallel_beam = list(cytoolz.concat(parallel_futures))
                prefix = ''
                # FIXME: maintain diversity in first-words here?
                beam = heapq.nlargest(beam_width, parallel_beam)
                # entry 2 is "DONE"
                if all(ent[2] for ent in beam):
                    break
            ents = [BeamEntry(*ent) for ent in beam]
            if len(ents) == 0:
                # Fall back on the full LM, but just for one word.
                first_word_ents = yield executor.submit(beam_search_phrases, domain, toks, beam_width=100, length_after_first=1, prefix_logprobs=prefix_logprobs, constraints=constraints)
                phrases = [(ent.words, None) for ent in first_word_ents[:3]]
            else:
                result = [ents.pop(0)]
                first_words = {ent.words[0] for ent in result}
                while len(result) < 3 and len(ents) > 0:
                    ents.sort(reverse=True, key=lambda ent: (ent.words[0] not in first_words, ent.score))
                    best = ents.pop(0)
                    first_words.add(best.words[0])
                    result.append(best)
                phrases = [([word for word in ent.words if word[0] != '<'], None) for ent in result]

        else: # sufarr

            # Use beam search on LM.
            if prefix_logprobs is None:
                sentence_enders = yield executor.submit(get_sentence_enders, domain, toks)
            else:
                sentence_enders = []

            beam_search_kwargs = dict(constraints=constraints)

            if polarity_split:
                clf = CLASSIFIERS['positive']
                clf_startstate = clf.get_state(toks)

            # Include a broader range of first words if we may need to diversify by sentiment after the fact.
            num_first_words = 3 - len(sentence_enders) if polarity_split is None else 10
            num_intermediates = 5
            max_logprob_penalty = -1.

            # Launch a job to get first words.
            if num_first_words:
                first_word_ents = yield executor.submit(beam_search_phrases,
                    domain, toks, beam_width=num_first_words, length_after_first=1, prefix_logprobs=prefix_logprobs, **beam_search_kwargs)
            else:
                first_word_ents = []

            first_words = {ent[1][0]: fwent_idx for fwent_idx, ent in enumerate(first_word_ents)}

            if promise is not None:
                promise_slot = promise['slot']
                # Remove the first word of the promise from the pool, we'll get to it later.
                promise_first_word = promise['words'][0]
                if promise_first_word in first_words:
                    first_word_ents.pop(first_words[promise_first_word])
            else:
                promise_slot = None

            jobs = [executor.submit(beam_search_phrases_loop, model, [ent],
                        start_idx=1,
                        beam_width=num_intermediates,
                        length_after_first=length_after_first, **beam_search_kwargs)
                    for ent in first_word_ents]
            if promise is not None and len(promise['words']) < 5:
                # Sneak an extra job into the queue...
                promise_extension = True
                # Promise provided, but we need to extend it with some new words.
                remaining_length = max(1, length_after_first - len(' '.join(promise['words'])))
                jobs.append(executor.submit(beam_search_phrases,
                    model, toks + promise['words'], beam_width=num_intermediates,
                    length_after_first=remaining_length, **beam_search_kwargs))
            else:
                promise_extension = False
            results = (yield jobs)
            if promise_extension:
                # Deal with the results of our extra job.
                extra_promise_words = results.pop()[0][1]
                promise = {'slot': promise_slot, 'words': promise['words'] + extra_promise_words}

            # Now build final suggestions.
            active_entities = []
            for beam in results:
                for ent in beam:
                    llk = ent[0]
                    words = ent[1]
                    pos = clf.classify_seq(clf_startstate, words) if polarity_split else None
                    active_entities.append((llk, pos, words, {}))

            # Add sentence-enders in the mix, but flagged special.
            for ender in sentence_enders[:2]:
                active_entities.append((995, .5, [ender], {'type': 'eos'}))

            # Add the promise continuation also, also flagged special.
            if promise is not None:
                llk = 999
                words = promise['words']
                pos = clf.classify_seq(clf_startstate, words) if polarity_split else None
                active_entities.append((llk, pos, words, {'type': 'promise'}))

            active_entities.sort(reverse=True)

            # Pad the active entities with null suggestions.
            for i in range(3):
                active_entities.append((-9999, .5, [''], {'type': 'null'}))

            entity_idx = 0
            promise_entity_idx = 0
            if promise is not None:
                # The zeroth entity should be the promise.
                assert active_entities[promise_entity_idx][3]['type'] == 'promise'
                # Start open-assignment at the first entity.
                entity_idx += 1

            # Take 3 suggestions
            assignments = [None] * 3
            first_words_used = {}
            for slot_idx in range(3):
                if slot_idx == promise_slot:
                    assignments[slot_idx] = promise_entity_idx
                    first_words_used[promise['words'][0]] = slot_idx
                    continue
                while True:
                    llk, pos, words, meta = active_entities[entity_idx]
                    first_word = words[0]
                    if first_word in first_words_used:
                        entity_idx += 1
                        continue
                    if first_word != '':
                        first_words_used[first_word] = slot_idx
                    assignments[slot_idx] = entity_idx
                    entity_idx += 1
                    break

            if polarity_split:
                print("First words:", ' '.join(ent[1][0] for ent in first_word_ents))

                # Diversify the suggestions by sentiment.
                cur_sentiments = np.array([active_entities[entity_idx][1] for entity_idx in assignments])
                cur_sentiment_diversity = scalar_diversity(cur_sentiments)
                min_logprob_allowed = min(active_entities[entity_idx][0] for entity_idx in assignments) + max_logprob_penalty

                # Greedily replace suggestions so as to increase sentiment diversity.
                while True:
                    for entity_idx in assignments:
                        llk, pos, words, meta = active_entities[entity_idx]
                        print(f"{pos:3.2f} {llk:6.2f} {' '.join(words)}")
                    print()
                    print()

                    candidates = []
                    for entity_idx, (llk, pos, words, meta) in enumerate(active_entities):
                        if llk < min_logprob_allowed:
                            continue
                        # Would this increase the sentiment diversity if we added it?
                        # Case 1: it replaces an existing word
                        replaces_slot = first_words_used.get(words[0])
                        if replaces_slot is not None:
                            prev_llk = active_entities[assignments[replaces_slot]][0]
                            if prev_llk >= 0:
                                # Sorry, this was a special one, can't kick it out.
                                # TODO: maybe we have a more diverse continuation for the promise?? check if it was a promise, and if so, what this starts with.
                                continue
                            candidate_sentiments = cur_sentiments.copy()
                            candidate_sentiments[replaces_slot] = pos
                            new_diversity = scalar_diversity(candidate_sentiments)
                        else:
                            # Case 2: it replaces the currently least-diverse word.
                            new_diversities = np.zeros(3)
                            for replaces_slot in range(3):
                                prev_llk = active_entities[assignments[replaces_slot]][0]
                                if prev_llk >= 0:
                                    # Sorry, this was a special one, can't kick it out.
                                    continue
                                candidate_sentiments = cur_sentiments.copy()
                                candidate_sentiments[replaces_slot] = pos
                                new_diversities[replaces_slot] = scalar_diversity(candidate_sentiments)
                            replaces_slot = np.argmax(new_diversities)
                            new_diversity = new_diversities[replaces_slot]
                        if new_diversity > cur_sentiment_diversity:
                            candidates.append((new_diversity, replaces_slot, entity_idx))
                    print(f"Found {len(candidates)} candidates that increase diversity")
                    if len(candidates) == 0:
                        break
                    prev_sentiment_diversity = cur_sentiment_diversity
                    cur_sentiment_diversity, replaces_slot, entity_idx = max(candidates)
                    llk, pos, words, meta = active_entities[entity_idx]
                    print(f"Replacing slot {replaces_slot} with llk={llk:.2f} pos={pos:.2f} \"{' '.join(words)}\" to gain {cur_sentiment_diversity - prev_sentiment_diversity:.2f} diversity")

                    # Actually replace the suggestion.
                    kicked_out_entity_idx = assignments[replaces_slot]

                    cur_sentiments[replaces_slot] = pos
                    existing_first_word = active_entities[kicked_out_entity_idx][2][0]
                    del first_words_used[existing_first_word]
                    first_words_used[words[0]] = replaces_slot
                    assignments[replaces_slot] = entity_idx

                if promise is not None:
                    assert assignments[promise_slot] == promise_entity_idx
                    assignments.pop(promise_slot)
                assignments.sort(key=lambda entity_idx: -active_entities[entity_idx][1])
                if promise is not None:
                    assignments.insert(promise_slot, promise_entity_idx)

            # Now we should have assignments of phrases to slots.
            phrases = []
            for entity_idx in assignments:
                llk, pos, words, meta = active_entities[entity_idx]
                phrases.append((words, dict(meta, llk=llk, pos=pos)))

    else:
        # TODO: upgrade to use_sufarr flag
        phrases = generate_diverse_phrases(
            domain, toks, 3, 6, prefix_logprobs=prefix_logprobs, temperature=temperature, use_sufarr=use_sufarr, **kw)
    return phrases, sug_state


def get_suggestions(*a, **kw):
    '''Wrap the async suggestion generation so it's testable.'''
    from concurrent.futures import Future
    class NullExecutor:
        def submit(self, fn, *a, **kw):
            future = Future()
            future.set_result(fn(*a, **kw))
            return future
    generator = get_suggestions_async(NullExecutor(), *a, **kw)
    result = None
    while True:
        try:
            result = generator.send(result)
            if isinstance(result, Future):
                result = result.result()
            elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], Future):
                result = [fut.result() for fut in result]
            else:
                print("Unexpected yield of something other than a Future!")
                return result
        except StopIteration as stop:
            return stop.value
