from suggestion import suggestion_generator

def collect_words_in_range_slow(start, after_end, word_idx):
    words = set()
    for idx in range(start, after_end):
        words.add(suggestion_generator.sufarr.get_suffix_by_idx(idx)[word_idx])
    return words


def test_collect_words_in_range():
    sufarr = suggestion_generator.sufarr
    start, end = sufarr.search_range(tuple('<D> my'.split()) + ('',))
    assert [] == suggestion_generator.collect_words_in_range(start, start, 1, sufarr.docs)

    ref = sorted(collect_words_in_range_slow(start, end, 1))
    assert ref == suggestion_generator.collect_words_in_range(start, end, 1, sufarr.docs)

    ref = sorted(collect_words_in_range_slow(start, end, 2))
    assert ref == suggestion_generator.collect_words_in_range(start, end, 2, sufarr.docs)

    import random
    rng = random.Random(0)
    for i in range(10):
        word_idx = rng.randrange(2, 4)
        suf = sufarr.get_suffix_by_idx(rng.randrange(0, len(suggestion_generator.sufarr.doc_idx)))[:word_idx]

        start, end = sufarr.search_range(tuple(suf) + ('',))
        ref = sorted(collect_words_in_range_slow(start, end, word_idx))
        assert ref == suggestion_generator.collect_words_in_range(start, end, word_idx, sufarr.docs)


DEFAULT_CONFIG = dict(domain='yelp_train', temperature=0.)
configs = dict(
    sufarr_rwb=dict(use_bos_suggs=False, rare_word_bonus=1.0, use_sufarr=True),
    sufarr_nobonus=dict(use_bos_suggs=False, rare_word_bonus=0.0, use_sufarr=True),
    unconstained_beamsearch=dict(use_bos_suggs=False, rare_word_bonus=None, use_sufarr=False),
    sufarr_and_bos=dict(use_bos_suggs=True, rare_word_bonus=2.0, use_sufarr=True),
    #dict(use_bos_suggs=False, rare_word_bonus=None, use_sufarr=False, temperature=.5)
)
configs = {k: dict(DEFAULT_CONFIG, **v) for k, v in configs.items()}

def test_get_suggestions():
    # A smoke test.
    for config in configs.values():
        result, new_sug_state = suggestion_generator.get_suggestions(
            sofar="one day , ",
            cur_word=[],
            **config)
        assert len(result) > 0
        phrase, probs = result[0]
        assert len(phrase) > 0
        assert isinstance(phrase[0], str)


def test_curword_with_no_followers():
    suggestion_generator.get_suggestions(
        sofar='some other foods since the ',
        cur_word=[{'letter': let} for let in 'servic'],
        domain='yelp_train',
        rare_word_bonus=1.0,
        use_sufarr=False,
        temperature=0.,
        use_bos_suggs=False)

def test_fallback_after_typo():
    result, new_sug_state = suggestion_generator.get_suggestions(
        sofar='thjs ', cur_word=[],
        **configs['sufarr_and_bos'])
    assert len(result) > 0


def test_odd_tokenization():
    suggestion_generator.tokenize_sofar('. ')
