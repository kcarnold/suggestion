from suggestion import suggestion_generator

def collect_words_in_range_slow(start, after_end, word_idx):
    words = set()
    for idx in range(start, after_end):
        words.add(suggestion_generator.sufarr.get_suffix_by_idx(idx)[word_idx])
    return words


def test_collect_words_in_range():
    start, end = suggestion_generator.sufarr.search_range(tuple('<D> my'.split()) + ('',))
    assert [] == suggestion_generator.collect_words_in_range(start, start, 1)

    ref = sorted(collect_words_in_range_slow(start, end, 1))
    assert ref == suggestion_generator.collect_words_in_range(start, end, 1)

    ref = sorted(collect_words_in_range_slow(start, end, 2))
    assert ref == suggestion_generator.collect_words_in_range(start, end, 2)
