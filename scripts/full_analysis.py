import re
import os
import argparse
import subprocess
import pandas as pd
import json

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
    any=r'Q_\w+|nextURL|clientId|Timing.*|Browser.*|Location.*|Recipient.*|Response.+|ExternalDataReference|Finished|Status|IPAddress|StartDate|EndDate|Welcome.+',
    )

prefix_subs = {
    "How much do you agree with the following statements about the suggestions that the system gave?-They ": "suggs-",
    "How much do you agree with the following statements?-The suggestions ": "suggs-",
    "Now think about the brainstorming you did before the final writing. How much do you agree with th...-": "brainstorm-",
    "Think about when you were typing out your ${e://Field/revisionDesc}. How much do you agree with t...-": "final-",
}

def run_log_analysis(participant):
    with open(os.path.join(root_path, 'logs', participant+'.jsonl')) as logfile:
        result = subprocess.check_output([os.path.join(root_path, 'frontend', 'analysis')], stdin=logfile)
    bundled_participants = json.loads(result)
    assert len(bundled_participants) == 1
    pid, analyzed = bundled_participants[0]
    return analyzed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('participants', nargs='+',
                        help='Participant ids to analyze')
    args = parser.parse_args()

    surveys = {name: pd.read_csv(
        os.path.join(root_path, 'surveys', name+'_responses.csv'),
        header=1, parse_dates=['StartDate', 'EndDate'])
        for name in survey_names}

    for participant in args.participants:
        log_analyses = run_log_analysis(participant)
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

