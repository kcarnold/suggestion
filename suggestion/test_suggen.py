from suggestion import suggestion_generator

def collect_words_in_range_slow(start, after_end, word_idx):
    words = set()
    for idx in range(start, after_end):
        words.add(suggestion_generator.sufarr.get_suffix_by_idx(idx)[word_idx])
    return words


def test_collect_words_in_range():
    sufarr = suggestion_generator.sufarr
    start, end = sufarr.search_range(tuple('<D> my'.split()) + ('',))
    assert [] == suggestion_generator.collect_words_in_range(start, start, 1)

    ref = sorted(collect_words_in_range_slow(start, end, 1))
    assert ref == suggestion_generator.collect_words_in_range(start, end, 1)

    ref = sorted(collect_words_in_range_slow(start, end, 2))
    assert ref == suggestion_generator.collect_words_in_range(start, end, 2)

    import random
    rng = random.Random(0)
    for i in range(10):
        word_idx = rng.randrange(2, 4)
        suf = sufarr.get_suffix_by_idx(rng.randrange(0, len(suggestion_generator.sufarr.doc_idx)))[:word_idx]

        start, end = sufarr.search_range(tuple(suf) + ('',))
        ref = sorted(collect_words_in_range_slow(start, end, word_idx))
        assert ref == suggestion_generator.collect_words_in_range(start, end, word_idx)



def test_get_suggestions():
    # A smoke test.
    configs = [
        dict(rare_word_bonus=1.0, use_sufarr=True, temperature=0.),
        dict(rare_word_bonus=0.0, use_sufarr=True, temperature=0.),
        dict(rare_word_bonus=None, use_sufarr=False, temperature=0.),
        #dict(rare_word_bonus=None, use_sufarr=False, temperature=.5)
        ]
    for config in configs:
        result = suggestion_generator.get_suggestions(
            sofar="one day , ",
            cur_word=[],
            domain='yelp_train',
            **config)
        assert len(result) > 0
        phrase, probs = result[0]
        assert len(phrase) > 0
        assert isinstance(phrase[0], str)


def test_curword_with_no_followers():
    suggestion_generator.get_suggestions(
        'some other foods since the ',
        [{'letter': let} for let in 'servic'],
        domain='yelp_train',
        rare_word_bonus=1.0,
        use_sufarr=False,
        temperature=0.)
