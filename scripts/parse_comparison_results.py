import pandas as pd
import json
from suggestion.paths import paths

def load_comparison_results():
    result_files = list(paths.parent.joinpath('gruntwork', 'turk_comparison_data').glob("Batch*results.csv"))
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
    res['rec_selected'] = res['rec_selected'].astype(float)
    return res

if __name__ == '__main__':
    res = load_comparison_results()
    count_chosen = res[~res.rec_selected.isnull()].groupby(['context', 'pairIdx', 'review_idx',
       'sent_idx', 'star_review', 'sugg', 'true_follows', 'word_idx']).rec_selected.agg(['count', 'sum']).reset_index()
    count_chosen['is_bos'] = count_chosen['word_idx'] == 0
    count_chosen.to_csv('comparison_annotation_parsed.csv', index=True)
