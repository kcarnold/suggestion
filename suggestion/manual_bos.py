import string
import numpy as np
import nltk
from suggestion.paths import paths


with open(paths.parent / 'clusters.txt') as f:
    groups = []
    group = []
    meta = None
    for line in f:
        line = line.strip()
        if line.startswith('='):
            meta = line[1:].strip()
            continue
        if not line:
            if group:
                groups.append((meta, group))
                group = []
            continue
        if line[0] in string.digits:
            continue
        group.append(line)
    if group:
        groups.append((meta, group))


def get_manual_bos(context, state):
    done_indices = state.get('done_indices', [])
    sent_idx = len(nltk.sent_tokenize(context))
    sugs = []
    for i in range(3):
        for j in range(100):
            group_idx = np.random.choice(len(groups))
            if group_idx in done_indices:
                continue
            meta, group = groups[group_idx]
            if sent_idx == 0 and meta not in ['START', 'EARLY']:
                continue
            if meta == "EARLY" and sent_idx > 1:
                continue
            if sent_idx == 9 and meta != "END":
                continue
            done_indices.append(group_idx)
            break
        else:
            print("Max tries reached")
            break
        sugs.append((np.random.choice(group).split(), {'bos': True}))
    # positivity = clf.classify_seq(clf.get_state([]), sent.split())
    # print(f'{positivity:.2f} {sent}')
    return sugs, state

