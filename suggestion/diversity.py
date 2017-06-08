import numpy as np


from scipy.spatial.distance import pdist, squareform

def scalar_diversity(x, max_distance=1.):
    x = np.array(x)[:,None]
    K = max_distance - squareform(pdist(x))
    K /= max_distance
    return np.linalg.det(K)


def get_suggs_with_sentiment_diversity(model, toks, *, clf, N_initial=10, N_post=5, max_logprob_penalty=-1.,
    promise=None, length_after_first=17, prefix_logprobs=None, constraints={}):

    from suggestion import suggestion_generator

    clf_startstate = clf.get_state(toks)

    # Pick first words
    first_word_ents = suggestion_generator.beam_search_phrases(model, toks,
        beam_width=N_initial, length_after_first=1, prefix_logprobs=prefix_logprobs, constraints=constraints)

    # Extend each first-word beam entity.
    ents = []
    for fwent in first_word_ents:
        beam = [fwent]
        for iteration_num in range(1, length_after_first):
            beam = suggestion_generator.beam_search_phrases_extend(model, beam, iteration_num=iteration_num, beam_width=N_post, length_after_first=length_after_first,
                prefix_logprobs=prefix_logprobs, rare_word_bonus=0., constraints=constraints)
            prefix_logprobs = None
        ents.extend(beam)

    classified_ents = [(ent[0], clf.classify_seq(clf_startstate, ent[1]), ent[1]) for ent in ents]
    classified_ents.sort(reverse=True)

    # Take 3 first-word suggestions.
    suggs = []
    first_words_used = {}
    for llk, pos, words in classified_ents:
        first_word = words[0]
        if first_word in first_words_used:
            continue
        first_words_used[first_word] = len(suggs)
        suggs.append((llk, pos, words))
        if len(suggs) == 3:
            break

    cur_sentiments = np.array([pos for llk, pos, words in suggs])
    cur_sentiment_diversity = scalar_diversity(cur_sentiments)
    min_logprob_allowed = min(llk for llk, pos, words in suggs) + max_logprob_penalty

    # Greedily replace suggestions so as to increase sentiment diversity.
    while True:
        for llk, pos, words in suggs:
            print(f"{pos:3.2f} {llk:6.2f} {' '.join(words)}")
        print()
        print()

        candidates = []
        for llk, pos, words in classified_ents:
            if llk < min_logprob_allowed:
                continue
            # Would this increase the sentiment diversity if we added it?
            # Case 1: it replaces an existing word
            replaces_slot = first_words_used.get(words[0])
            if replaces_slot is not None:
                candidate_sentiments = cur_sentiments.copy()
                candidate_sentiments[replaces_slot] = pos
                new_diversity = scalar_diversity(candidate_sentiments)
            else:
                # Case 2: it replaces the currently least-diverse word.
                new_diversities = np.empty(3)
                for replaces_slot in range(3):
                    candidate_sentiments = cur_sentiments.copy()
                    candidate_sentiments[replaces_slot] = pos
                    new_diversities[replaces_slot] = scalar_diversity(candidate_sentiments)
                replaces_slot = np.argmax(new_diversities)
                new_diversity = new_diversities[replaces_slot]
            if new_diversity > cur_sentiment_diversity:
                candidates.append((new_diversity, replaces_slot, llk, pos, words))
        print(f"Found {len(candidates)} candidates that increase diversity")
        if len(candidates) == 0:
            break
        prev_sentiment_diversity = cur_sentiment_diversity
        cur_sentiment_diversity, replaces_slot, llk, pos, words = max(candidates)
        existing_sugg = suggs[replaces_slot]
        print(f"Replacing slot {replaces_slot} with llk={llk:.2f} pos={pos:.2f} \"{' '.join(words)}\" to gain {cur_sentiment_diversity - prev_sentiment_diversity:.2f} diversity")

        # Actually replace the suggestion.
        cur_sentiments[replaces_slot] = pos
        del first_words_used[existing_sugg[2][0]]
        first_words_used[words[0]] = replaces_slot
        suggs[replaces_slot] = (llk, pos, words)

    return [(words, dict(llk=llk, pos=pos)) for llk, pos, words in suggs]
