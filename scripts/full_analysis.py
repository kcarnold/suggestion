import re
import os
import argparse
import subprocess
import pandas as pd
import json
import numpy as np

root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

survey_names =  ['intro', 'postFreewrite', 'postTask', 'postExp']
#'instructionsQuiz',

survey_seq = '''
intro 0
postFreewrite 0
postTask 0
postFreewrite 1
postTask 1
postExp 0
'''.strip().split('\n')
survey_seq = (x.split(' ') for x in survey_seq)
survey_seq = [(name, int(idx)) for name, idx in survey_seq]

skip_col_re = dict(
    any=r'Q_\w+|nextURL|clientId|Timing.*|Browser.*|Location.*|Recipient.*|Response.+|ExternalDataReference|Finished|Status|IPAddress|StartDate|EndDate|Welcome.+|Display Order',
    )

prefix_subs = {
    "How much do you agree with the following statements about the suggestions that the system gave?-They ": "suggs-",
    "How much do you agree with the following statements?-The suggestions ": "suggs-",
    "Now think about the brainstorming you did before the final writing. How much do you agree with th...-": "brainstorm-",
    "Think about when you were typing out your ${e://Field/revisionDesc}. How much do you agree with t...-": "final-",
    "How Accurately Can You Describe Yourself? Describe yourself as you generally are now, not as you...-": "pers-",
}

def run_log_analysis(participant):
    with open(os.path.join(root_path, 'logs', participant+'.jsonl')) as logfile:
        result = subprocess.check_output([os.path.join(root_path, 'frontend', 'analysis')], stdin=logfile)
    bundled_participants = json.loads(result)
    assert len(bundled_participants) == 1
    pid, analyzed = bundled_participants[0]
    return analyzed

#%%
import random
def split_randomly_without_overlap(total_num_items, chunk_size, views_per_item, rs):
    remaining_views = [views_per_item] * total_num_items
    chunks = []
    while sum(remaining_views) >= chunk_size:
        chunk = []
        for i in range(chunk_size):
            mrv = max(remaining_views)
            opts = [i for i, rv in enumerate(remaining_views) if rv == mrv and i not in chunk]
#            item = np.argmax(remaining_views)
            item = rs.choice(opts)
            assert item not in chunk
            chunk.append(item)
            remaining_views[item] -= 1
        chunks.append(chunk)
    return chunks
split_randomly_without_overlap(10, 4, 3, rs=random.Random(0))
#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('participants', nargs='+',
                        help='Participant ids to analyze')
    args = parser.parse_args()

    surveys = {name: pd.read_csv(
        os.path.join(root_path, 'surveys', name+'_responses.csv'),
        header=1, parse_dates=['StartDate', 'EndDate'])
        for name in survey_names}

    all_log_analyses = {}
    participants = args.participants
    assert len(participants) == len(set(participants))

    for participant in participants:
        log_analyses = run_log_analysis(participant)
        all_log_analyses[participant] = log_analyses
        survey_data = {name: survey[survey['clientId'] == participant].to_dict(orient='records')#.to_json(orient='records'))
            for name, survey in surveys.items()}
        with open(os.path.join(root_path, 'logs', participant+'-analyzed.json'), 'w') as f:
            json.dump(dict(log_analyses=log_analyses, survey_data=survey_data), f)

        print('\n'*10)
        print(participant)
        print('='*80)
        skipped_cols = set()
        for survey, idx in survey_seq:
            print(f"\n\n{survey} {idx}")
            try:
                data = survey_data[survey][idx]
            except IndexError:
                print("MISSING!?!?!")
                continue
            print('-'*20)
            for k, v in data.items():
#                if np.isnan(v):
#                    continue
                if re.match(skip_col_re['any'], k):
                    skipped_cols.add(k)
                    continue
                for x, y in prefix_subs.items():
                    if k.startswith(x):
                        k = k.replace(x, y, 1)
                        break
                print(k)
                print(v)
                print()

    CHUNK_SIZE = 4
    VIEWS_PER_ITEM = 3

    splits = split_randomly_without_overlap(len(args.participants), CHUNK_SIZE, VIEWS_PER_ITEM, rs=random.Random(0))
    data = [{
            "pages": [[
                    dict(participant_id=participants[idx], cond=block['condition'], text=block['finalText']) for block in all_log_analyses[participants[idx]]['blocks']]
                    for idx in chunk],
            "attrs": ["food", "drinks", "atmosphere", "service", "value"],
            } for chunk in splits]

    pd.DataFrame(dict(data=[json.dumps(d) for d in data])).to_csv(f'analyzed_{"_".join(args.participants)}.csv', index=False)
