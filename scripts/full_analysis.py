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
    any=r'Great.job|Q_\w+|nextURL|clientId|Timing.*|Browser.*|Location.*|Recipient.*|Response.+|ExternalDataReference|Finished|Status|IPAddress|StartDate|EndDate|Welcome.+|Display Order',
    )

prefix_subs = {
    "How much do you agree with the following statements about the suggestions that the system gave?-They ": "suggs-",
    "How much do you agree with the following statements?-The suggestions ": "suggs-",
    "Now think about the brainstorming you did before the final writing. How much do you agree with th...-": "brainstorm-",
    "Think about when you were typing out your ${e://Field/revisionDesc}. How much do you agree with t...-": "final-",
    "How Accurately Can You Describe Yourself? Describe yourself as you generally are now, not as you...-": "pers-",
}

decode_scales = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Somewhat disagree": 3,
        "Neither agree nor disagree": 4,
        "Somewhat agree": 5,
        "Agree": 6,
        "Strongly agree": 7,

        "Very Inaccurate": 1,
        "Moderately Inaccurate": 2,
        "Neither Accurate Nor Inaccurate": 3,
        "Moderately Accurate": 4,
        "Very Accurate": 5}

participants = dict(study2='''81519e
81f6b6
9b6cd1
d9100a
852f7a
83fa09
f3542e
4f99b7
c5c40b
88d3ad
0cb74f
f31d92
4edc26
885dae
a997ed
8c01ef
773fa0
43cd2c
706d74
7d5d97'''.split(), study3='''4265fc 6e3526 15b070 a10da3 6c0f8a'''.split())['study3']


def run_log_analysis(participant):
    logpath = os.path.join(root_path, 'logs', participant+'.jsonl')
    with open(logpath) as logfile:
        result = subprocess.check_output([os.path.join(root_path, 'frontend', 'analysis')], stdin=logfile)
    bundled_participants = json.loads(result)
    assert len(bundled_participants) == 1
    pid, analyzed = bundled_participants[0]
    with open(logpath) as logfile:
        lines = (json.loads(line) for line in logfile)
        analyzed['sug_gen_durs'] = [rec['msg']['dur'] for rec in lines if rec.get('type') == 'receivedSuggestions']

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



#%%

def classify_annotated_event(evt):
    text = evt['curText']
    null_word = len(text) == 0 or text[-1] == ' '
    typ = evt['type']
    if typ == 'tapKey':
        return 'tapKey'
    if typ == 'tapBackspace':
        return 'tapBackspace'
    if typ == 'tapSuggestion':
        return 'tapSugg_' + ('full' if null_word else 'part')
    assert typ

from collections import Counter

if __name__ == '__main__':

    surveys = {name: pd.read_csv(
        os.path.join(root_path, 'surveys', name+'_responses.csv'),
        header=1, parse_dates=['StartDate', 'EndDate'])
        for name in survey_names}

    all_log_analyses = {}
#    participants = args.participants
    assert len(participants) == len(set(participants))

    all_survey_data = []
    participant_level_data = []
    excluded = []
    non_excluded_participants = []
    for participant in participants:
        log_analyses = run_log_analysis(participant)
        all_log_analyses[participant] = log_analyses
        conditions = log_analyses['conditions']
        survey_data = {name: survey[survey['clientId'] == participant].to_dict(orient='records')#.to_json(orient='records'))
            for name, survey in surveys.items()}
        with open(os.path.join(root_path, 'logs', participant+'-analyzed.json'), 'w') as f:
            json.dump(dict(log_analyses=log_analyses, survey_data=survey_data), f)

        if len(survey_data['postExp']) == 0:
            # Skip incomplete experiments.
            excluded.append((participant, 'incomplete'))
            continue

        datum = dict(participant_id=participant,
                     conditions=','.join(conditions),
                     config=log_analyses['config'])
        datum['dur_75'] = np.percentile(log_analyses['sug_gen_durs'], 75)
        datum['dur_95'] = np.percentile(log_analyses['sug_gen_durs'], 95)
        for page, page_data in log_analyses['byExpPage'].items():
            for typ, count in Counter(classify_annotated_event(evt) for evt in page_data['annotated']).items():
                datum[f'{page}_num_{typ}'] = count

        if datum['dur_75'] > .75:
            excluded.append(participant)
            continue
        participant_level_data.append(datum)

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
            base_data = {}
            if survey in ['postFreewrite', 'postTask']:
                base_data['condition'] = conditions[idx]
            elif survey == 'postExp':
                base_data['condition'] = ','.join(conditions)
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
                v = decode_scales.get(v, v)

                if k.startswith("Which writing was..."):
                    v = v.replace("A", conditions[0]).replace("B", conditions[1])
                elif survey == 'postExp' and isinstance(v, str):
                    v = re.sub(r'\bA\b', f'A [{conditions[0]}]', v)
                    v = re.sub(r'\bB\b', f'B [{conditions[1]}]', v)
                print(k)
                print(v)
                print()
                all_survey_data.append(dict(base_data,
                        participant_id=participant, survey=survey, idx=idx, name=k, value=v))
        non_excluded_participants.append(participant)


    pd.DataFrame(all_survey_data).to_csv(f'all_survey_data_{"_".join(participants)}.csv', index=False)
    pd.DataFrame(participant_level_data).to_csv(f'participant_level_data_{"_".join(participants)}.csv', index=False)
    print('excluded:', excluded)

#%%
if False:
    CHUNK_SIZE = 4
    VIEWS_PER_ITEM = 3
    splits = split_randomly_without_overlap(len(participants), CHUNK_SIZE, VIEWS_PER_ITEM, rs=random.Random(0))
    data = [{
            "pages": [[
                    dict(participant_id=participants[idx], cond=block['condition'], text=block['finalText']) for block in all_log_analyses[participants[idx]]['blocks']]
                    for idx in chunk],
            "attrs": ["food", "drinks", "atmosphere", "service", "value"],
            } for chunk in splits]

    pd.DataFrame(dict(data=[json.dumps(d) for d in data])).to_csv(f'analyzed_{"_".join(participants)}.csv', index=False)


#%%
    to_rate = [['852f7a', '88d3ad', '0cb74f', 'f31d92'], ['4edc26', '885dae', 'a997ed', '8c01ef'], ['773fa0', '43cd2c', '706d74', '7d5d97']]
    data = [{
            "pages": [[
                    dict(participant_id=participant_id, cond=block['condition'], text=block['finalText']) for block in all_log_analyses[participant_id]['blocks']]
                    for participant_id in chunk],
            "attrs": ["food", "drinks", "atmosphere", "service", "value"],
            } for chunk in to_rate]

    import datetime
    pd.DataFrame(dict(data=[json.dumps(d) for d in data])).to_csv(f'to_rate_{datetime.datetime.now().isoformat()}.csv', index=False)
