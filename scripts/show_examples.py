import nltk
from suggestion import suggestion_generator
#%%
from suggestion.suggestion_generator import collect_words_in_range, LOG10, get_model, tokenize_sofar
from collections import namedtuple
BeamEntry = namedtuple("BeamEntry", 'score, words, done, penultimate_state, last_word_idx, num_chars, rarest_word, rarest_freq')

import heapq
import kenlm
def beam_search_sufarr(model, sufarr, start_words, beam_width, length, rare_word_bonus=0., prefix=''):
    unigram_probs = suggestion_generator.get_unigram_probs(model)
    start_state, start_score = model.get_state(start_words[:-1], bos=False)
    beam = [(0., [], False, start_state, model.model.vocab_index(start_words[-1]), 0, None, 0.)]
    stats = []
    for i in range(length):
        prefix_chars = 1 if i > 0 else 0
        def candidates():
            for entry in beam:
                score, words, done, penultimate_state, last_word_idx, num_chars, rarest_word, rarest_score = entry
                if done:
                    yield entry
                    continue
                last_state = kenlm.State()
                model.model.base_score_from_idx(penultimate_state, last_word_idx, last_state)
                start_idx, end_idx = sufarr.search_range((start_words[-1],) + tuple(words) + (prefix,))
                next_words = collect_words_in_range(start_idx, end_idx, i + 1)
                stats.append((end_idx - start_idx, len(next_words)))
                if len(next_words) == 0:
                    assert model.id2str[last_word_idx] == '</S>', "We only expect to run out of words at an end-of-sentence that's also an end-of-document."
                    continue
                new_state = kenlm.State()
                bigram_prev_state = kenlm.State()
                bigram_tmp_state = kenlm.State()
                model.model.base_score_from_idx(model.null_context_state, last_word_idx, bigram_prev_state)
                for next_idx, word in enumerate(next_words):
#                    is_punct = word[0] in '<.!?'
                    word_idx = model.model.vocab_index(word)
                    if word_idx == 0:
                        # Unknown word, skip.
                        continue
                    is_special = word[0] == '<'
                    if is_special:
                        continue
                    new_words = words + [word]
                    new_num_chars = num_chars + (0 if is_special else prefix_chars + len(word))
                    logprob = LOG10 * model.model.base_score_from_idx(last_state, word_idx, new_state)
                    unigram_prob = unigram_probs[word_idx]
                    bigram_score = LOG10 * model.model.base_score_from_idx(bigram_prev_state, word_idx, bigram_tmp_state)
#                    rarity_score = -unigram_prob
                    rarity_score = -bigram_score
                    new_score = score + logprob
                    if not is_special and rarity_score > rarest_score:
                        new_score += rare_word_bonus * (rarity_score - rarest_score)
                        new_rarest_word = word
                        new_rarest_score = rarity_score
                    else:
                        new_rarest_word = rarest_word
                        new_rarest_score = rarest_score
                    done = new_num_chars >= length
                    yield BeamEntry(new_score, new_words, done, last_state, word_idx, new_num_chars, new_rarest_word, new_rarest_score)
        beam = heapq.nlargest(beam_width, candidates())
        prefix = ''
    # nlargest guarantees that its result is sorted descending.
    # print(stats)
    return [BeamEntry(*ent) for ent in beam]

#%%
example_doc = open('example_doc.txt').read()
words = nltk.word_tokenize(example_doc)
#%%
LENGTH = 30
conds = dict(
    A=dict(rare_word_bonus=1.0, useSufarr=True, temperature=0.),
    B=dict(rare_word_bonus=0.0, useSufarr=True, temperature=0.)
    )

model = get_model('yelp_train')
for i in range(0, 50):
    prefix = ' '.join(words[:i])
    print()
    print(prefix[-50:])

    for name, cond in conds.items():
        phrases = suggestion_generator.get_suggestions(
                    prefix + ' ', [],
                    domain='yelp_train',
                    rare_word_bonus=cond.get('rare_word_bonus', 1.0),
                    use_sufarr=cond.get('useSufarr', False),
                    temperature=cond.get('temperature', 0.),
                    length=LENGTH)
        print(' {:8}: {}'.format(name, ' '.join(phrases[0][0])))

    toks = tokenize_sofar(prefix + ' ')
    new_phrases = beam_search_sufarr(model, suggestion_generator.sufarr, toks, beam_width=100, length=LENGTH, rare_word_bonus=1.)
    print(' {:8}: {} ({}={:.2f})'.format('new', ' '.join(new_phrases[0].words), new_phrases[0].rarest_word, new_phrases[0].rarest_freq))
#    print(' {:8}: {}'.format('C', ' '.join(new_phrases[0].words)))

