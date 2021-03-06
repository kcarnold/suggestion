import json
import re
import tqdm
import pandas as pd
import numpy as np
from suggestion.paths import paths

all_reviews = pd.read_pickle(str(paths.parent / 'yelp_preproc' / 'valid_data.pkl'))['data']

reviews_by_stars = list(all_reviews.groupby('stars_review'))

#%% Pretend to retype existing reviews, look at suggestions offered.
from suggestion import suggestion_generator

flags = dict(
    domain='yelp_train-balanced',
    temperature=0.,
    rare_word_bonus=0.,
    continuation_length=17,
    use_sufarr=False,
    use_bos_suggs=False)

def untokenize(s):
    # s = re.sub(r'\bi\b', 'I', s)
    return re.sub(r'(\w) (\W)', r'\1\2', s)

NUM_COMPARISONS = 500

rs = np.random.RandomState(2)
samples = []
#%%
progress = tqdm.tqdm(total=NUM_COMPARISONS)
progress.update(len(samples))
star = 0
MIN_WORDS_AT_START = 1
PHRASE_LEN = 5
NUM_TO_SHOW = 1
while len(samples) < NUM_COMPARISONS:
    star_review, reviews_with_star = reviews_by_stars[star]
    review_idx = rs.choice(len(reviews_with_star))
    review = reviews_with_star.iloc[review_idx]
    sents = [sent for sent in review.tokenized.lower().split('\n')]
    sent_idx = rs.choice(len(sents))
    sent_toks = sents[sent_idx].split()
    sent_len = len(sent_toks)
    if sent_len < MIN_WORDS_AT_START + PHRASE_LEN + 1: # make sure we have at least one word to start, and don't count final punct
        continue
    use_bos = rs.random_sample() < .5
    if use_bos:
        word_idx = 0
    else:
        word_idx = rs.randint(MIN_WORDS_AT_START, sent_len - PHRASE_LEN)
    context = ' '.join((' '.join(sents[:sent_idx]).split() + sent_toks[:word_idx])[-6:])
    true_follows = sent_toks[word_idx:][:PHRASE_LEN]
    assert len(true_follows) >= PHRASE_LEN
    # try:
    suggestions = [phrase for phrase, probs in suggestion_generator.get_suggestions(
        sofar=context+' ', cur_word=[], **flags)[0]]
    # except Exception:
    #     continue
    sug_idx = rs.choice(len(suggestions))
    picked_sugg = suggestions[sug_idx]
    picked_sugg = picked_sugg[:NUM_TO_SHOW]
    true_follows = true_follows[:NUM_TO_SHOW]
    if picked_sugg == true_follows:
        continue
    samples.append(dict(
            star_review=star_review,
            review_idx=review_idx,
            sent_idx=sent_idx,
            word_idx=word_idx,
            context=untokenize(context),
            true_follows=' '.join(true_follows),
            sugg=' '.join(picked_sugg)))
    progress.update(1)
    star += 1
    star = star % len(reviews_by_stars)

json.dump(samples, open(paths.parent / 'gruntwork' / f'comparisons_existing_reviews_{NUM_TO_SHOW}words.json', 'w'), default=lambda x: x.tolist())
