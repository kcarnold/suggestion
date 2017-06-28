import os
import pandas as pd
from collections import Counter

from suggestion.paths import paths
root_path = paths.parent

from suggestion.util import mem, flatten_dict

from suggestion.analysis_util import (
        # survey stuff
        skip_col_re, prefix_subs,
        # log analysis stuff
        get_existing_requests, classify_annotated_event, get_log_analysis)


STUDY_COLUMNS = '''
experiment_name
config
rev
conditions
instructions'''.strip().split()

PARTICIPANT_LEVEL_COLUMNS='''
participant_id
age
gender
education
english_proficiency
verbalized_during
'''.strip().split()

TRIAL_COLUMNS = '''
block
condition
latency_75
know_what_to_write
stars_before
stars_after
self_report_accuracy
final_text
num_tapBackspace
num_tapKey
num_tapSugg_bos
num_tapSugg_full
num_tapSugg_part
total_time
'''.strip().split()

#expression_{knewWhat,knewHow,pausedWhat,pausedHow,expressFluent,troubleExpressFluent}
#sugg_opinion_{relevant,interesting,...}

ANALYSIS_COLUMNS = '''
is_excluded
final_length_chars
corrected_text
final_length_words
num_sentences
unigram_llk_mean
unigram_llk_std
contextual_llk_mean
contextual_llk_std
total_positive
total_negative
max_positive
max_negative
num_topics
mtld
pairdist_words_mean
pairdist_words_std
pairdist_sentences_mean
pairdist_sentences_std
'''.strip().split()

VALIDATION_COLUMNS = '''
sugg_unigram_llk_mean
sugg_unigram_llk_std
sugg_sentiment_mean
sugg_sentiment_std
sugg_sentiment_group_std_mean
'''.strip().split()


@mem.cache
def get_survey_data_raw():
    survey_names = ['intro', 'postTask', 'postTask3', 'postExp', 'postExp3']
    # TODO: use the prewrites too?
    # .iloc[1:] is to skip the ImportID row.
    return {name: pd.read_csv(
        os.path.join(root_path, 'surveys', name+'_responses.csv'),
        header=1, parse_dates=['StartDate', 'EndDate']).iloc[1:]
        for name in survey_names}


def process_survey_data(survey_data_raw):
    data = pd.concat(survey_data_raw, names=['survey', 'survey_record_idx']).reset_index(level=-1, drop=True).reset_index()
    data = data.rename(columns={'clientId': 'participant_id'})
    data['survey_trial_idx'] = data.groupby(['participant_id', 'survey']).cumcount()
    data = data.dropna(subset=['participant_id'])

    # Drop junk columns.
    cols_to_drop = [col for col in data.columns if skip_col_re.match(col)]
#    print(cols_to_drop)
    data = data.drop(cols_to_drop, axis=1)

    # Bulk renames
    cols_to_rename = {}
    for col in data.columns:
        for x, y in prefix_subs.items():
            if col.startswith(x):
                cols_to_rename[col] = col.replace(x, y, 1)
                break
    data = data.rename(columns=cols_to_rename)

    # Specific renames
    renames = {
        "How old are you?": ("age", 'numeric'),
        "What is your gender?": ("gender", None),
        "Now that you've had a chance to write about it, how many stars would you give your experience at...-&nbsp;": ("stars_after", 'numeric'),
        "How proficient would you say you are in English?": ("english_proficiency", None),
        "What is the highest level of school you have completed or the highest degree you have received? ": ("education", None),
        "While you were writing, did you speak or whisper what you were writing?": ("verbalized_during", None),
    }
    for orig, new in renames.items():
        col_data = data.pop(orig)
        new_name = new[0]
        if new[1] == 'numeric':
            col_data = pd.to_numeric(col_data)
        data[new_name] = col_data
    return data


def get_participants_by_study():
    import yaml
    participants_table = []
    for study_name, participants in yaml.load(open(root_path / 'participants.yaml')).items():
        for participant in participants.split():
            participants_table.append((participant, study_name))
    return pd.DataFrame(participants_table, columns=['participant_id', 'study'])


def get_log_analysis_data(participant):
    participant_level_data_raw = []
    log_analyses = get_log_analysis(participant)
    conditions = log_analyses['conditions']
    base_datum = dict(participant_id=participant,
                 conditions=','.join(conditions),
                 config=log_analyses['config'])
    for page, page_data in log_analyses['byExpPage'].items():
        datum = base_datum.copy()
        if '-' in page:
            kind, num = page.split('-')
        else:
            kind = page
            num = '0'
        num = int(num)
        datum['kind'] = kind
        if kind != 'final':
            # FIXME: maybe in the future we'll want to look at practice and prewrite data too?
            continue
        datum['block'] = num
        datum.update(flatten_dict(log_analyses['blocks'][num]))
        renames = {
            'finalText': 'final_text',
            'place_knowWhatToWrite': 'know_what_to_write',
            'place_stars': 'stars_before'}
        for old_name, new_name in renames.items():
            datum[new_name] = datum.pop(old_name)
        classified_events = [classify_annotated_event(evt) for evt in page_data['annotated']]
#        transitions = Counter(zip(itertools.chain(['start'], classified_events), itertools.chain(classified_events, ['end']))).most_common()
#        for (a, b), count in transitions:
#            if a is not None and b is not None:
#                datum[f'x_{a}_{b}'] = count
        for typ, count in Counter(classified_events).items():
            if typ is not None:
                datum[f'num_{typ}'] = count
        participant_level_data_raw.append(datum)
    return participant_level_data_raw

#%%
#@mem.cache
def get_all_data():
    participants_by_study = get_participants_by_study()

    survey_data = process_survey_data(get_survey_data_raw())
#    survey_data = survey_data.
    is_participant_level = survey_data.survey.isin(['intro', 'postExp', 'postExp3'])
    participant_level_survey_data = survey_data[is_participant_level].dropna(axis=1, thresh=10).drop(['survey_trial_idx'], axis=1)
    trial_level_survey_data = survey_data[~is_participant_level].dropna(axis=1, thresh=10).rename(columns={'survey_trial_idx': 'block'})

    participant_level_data = pd.merge(
            participant_level_survey_data, participants_by_study,
            left_on='participant_id', right_on='participant_id', how='outer')

    # FIXME: remove the flag
    log_analysis_data_raw = {participant: get_log_analysis_data(participant)
        for participant in participants_by_study.participant_id}
    log_analysis_data = pd.concat({participant: pd.DataFrame(data) for participant, data in log_analysis_data_raw.items()}).reset_index(drop=True)

    trial_level_data = pd.merge(
            trial_level_survey_data,
            log_analysis_data,
            left_on=['participant_id', 'block'], right_on=['participant_id', 'block'], how='outer')

    full_data = pd.merge(
            participant_level_data,
            trial_level_data,
            left_on='participant_id', right_on='participant_id', how='right')

    desired_cols = set(STUDY_COLUMNS + PARTICIPANT_LEVEL_COLUMNS + TRIAL_COLUMNS + ANALYSIS_COLUMNS + VALIDATION_COLUMNS)
    missing_cols = sorted(desired_cols - set(full_data.columns))
    extra_cols = sorted(set(full_data.columns) - desired_cols)
    print(f"Missing {len(missing_cols)} cols", missing_cols)
    print(f"{len(extra_cols)} extra cols", extra_cols)
    return full_data

all_data = get_all_data()

