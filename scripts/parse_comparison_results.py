import pandas as pd
import json
from suggestion.paths import paths

def load_comparison_results(basename='turk_comparison_data'):
    result_files = list(paths.parent.joinpath('gruntwork', basename).glob("Batch*results.csv"))
    raw = pd.concat([pd.read_csv(str(f)) for f in result_files], axis=0, ignore_index=True)
    records = raw.loc[:, ['WorkerId', 'Answer.results']].to_dict('records')
    res = []
    for record in records:
        worker_id = record['WorkerId']
        annos = json.loads(record['Answer.results'])
        for entry in annos:
            left_is_rec = entry['left'] == entry['sugg']
            right_is_rec = entry['right'] == entry['sugg']
            assert left_is_rec or right_is_rec
            vote = entry['selected']
            if not vote:
                print(f"Missing: {worker_id} {entry['context']}")
                continue
            if vote == 'left':
                rec_selected = left_is_rec
            elif vote == 'right':
                rec_selected = right_is_rec
            else:
                assert vote == 'neither'
                rec_selected = None
            res.append(dict(entry, worker_id=worker_id, rec_selected=rec_selected))
    res = pd.DataFrame(res)
    # res['rec_selected'] = res['rec_selected'].astype(float)
    return res

def aggregate_selected(group):
    from collections import Counter
    if hasattr(group, 'tolist'):
        group = group.tolist()
    counts = Counter(group)
    if counts[True] > counts[None] and counts[True] > counts[False]:
        return 1
    if counts[False] > counts[None] and counts[False] > counts[True]:
        return -1
    return 0

for tester, proper in [
    ([True] * 10, 1),
    ([False] * 10, -1),
    ([None] * 10, 0),
    ([True] * 5 + [False] * 5, 0),
    ([True] * 2 + [None] * 2, 0),
    ([True] * 2 + [None] * 1 + [False] * 2, 0)]:
    print(f'{tester}: {aggregate_selected(tester)}')
    assert aggregate_selected(tester) == proper

if __name__ == '__main__':
    basename = 'turk_comparison_data_1word'
    res = load_comparison_results(basename=basename)
    count_chosen = res.groupby(['context', 'pairIdx', 'review_idx',
       'sent_idx', 'star_review', 'sugg', 'true_follows', 'word_idx']).rec_selected.agg(aggregate_selected).reset_index()
    count_chosen['is_bos'] = count_chosen['word_idx'] == 0
    count_chosen['rec_won'] = (count_chosen['rec_selected'] > 0).astype(float)
    count_chosen.to_csv(f'{basename}_parsed.csv', index=True)
