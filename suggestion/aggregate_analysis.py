import os
import pandas as pd
import numpy as np
import json
import subprocess
from collections import Counter
import nltk

from suggestion.paths import paths
root_path = paths.parent

from suggestion.util import mem, flatten_dict
from suggestion import tokenization
import string

from suggestion.analysis_util import (
        # survey stuff
        skip_col_re, prefix_subs, decode_scales,
        # log analysis stuff
        get_existing_requests, classify_annotated_event, get_log_analysis)


from suggestion.analyzers import WordFreqAnalyzer
analyzer = WordFreqAnalyzer.build()

ALL_SURVEY_NAMES = ['intro', 'intro2', 'postTask', 'postTask3', 'postExp', 'postExp3', 'postExp4']


STUDY_COLUMNS = '''
experiment_name
config
git_rev
conditions
instructions'''.strip().split()

PARTICIPANT_LEVEL_COLUMNS='''
participant_id
age
gender
education
english_proficiency
verbalized_during
total_actions_nobackspace
total_key_taps
total_rec_taps
rec_frac
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
    # TODO: use the prewrites too?
    # .iloc[1:] is to skip the ImportID row.
    return {name: pd.read_csv(
        os.path.join(root_path, 'surveys', name+'_responses.csv'),
        header=1, parse_dates=['StartDate', 'EndDate']).iloc[1:]
        for name in ALL_SURVEY_NAMES}


def process_survey_data(survey, survey_data_raw):
    is_repeated = survey in ['postTask', 'postTask3', 'postFreewrite']
    data = survey_data_raw
    data = data.rename(columns={'clientId': 'participant_id'})
    if is_repeated:
        data['block'] = data.groupby(['participant_id']).cumcount()
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

    data = data.applymap(lambda x: decode_scales.get(x, x))

    # Specific renames
    if survey in ['intro', 'intro2']:
        renames = {
            "How old are you?": ("age", 'numeric'),
            "What is your gender?": ("gender", None),
            "How proficient would you say you are in English?": ("english_proficiency", None),
            "What is the highest level of school you have completed or the highest degree you have received? ": ("education", None),
            "About how many online reviews (of restaurants or otherwise) have you written in the past 3 months?": ("reviewing_experience", None),
        }
    if survey in ['postTask', 'postTask3']:
        renames = {
            "Now that you've had a chance to write about it, how many stars would you give your experience at...-&nbsp;": ("stars_after", 'numeric'),
            "Compared with the experience you were writing about, the phrases that the keyboard gave were usua...": ("sentiment_manipcheck_posttask", None),
        }
    if survey.startswith('postExp'):
        renames = {
            "While you were writing, did you speak or whisper what you were writing?": ("verbalized_during", None),
        }
    for orig, new in renames.items():
        if orig not in data.columns:
            continue
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
    return pd.DataFrame(participants_table, columns=['participant_id', 'study']).drop_duplicates(subset=['participant_id'])


def get_correct_git_revs():
    '''
    Logs prior to 2017-06-30 had the git revision of the last time the server
    restarted, not actually the revision of the code.

    Let's assume that, while I may have continued coding after launching the study,
    that at least the _first_ log in the batch has the right date. So get that date, and the corresponding commit.

    Now, what if we had a collision, but I didn't notice because only one of the people continued the study?
    Then we'd have a really wrong date based on looking at the first one. But maybe there's few enough that I can spot-check them.

    I do know that all of the studies that are in 'participants' have completed writings.
    But they may have been sitting there reconnecting on the page for a long time afterwards.
    '''

    participants_by_study = get_participants_by_study()
    data = []
    for participant in participants_by_study.itertuples():
        participant_id = participant.participant_id
        logpath = paths.parent / 'logs' / (participant_id+'.jsonl')
        with open(logpath) as logfile:
            for line in logfile:
                line = json.loads(line)
                if line['type'] == 'connected':
                    data.append((participant_id, participant.study, line['rev'], line['timestamp']))
                    break
    revs_and_timestamps = pd.DataFrame(data, columns=['participant_id', 'study', 'git_rev', 'timestamp'])
    revs_and_timestamps['timestamp'] = pd.to_datetime(revs_and_timestamps['timestamp'])
    min_timestamps = revs_and_timestamps.drop_duplicates(subset=['participant_id']).groupby(['study']).timestamp.min()
    def get_commit_at_timestamp(ts):
        return subprocess.check_output(['git', 'describe', '--always', 'master@{'+ str(ts) + '}']).decode('latin1').strip()
    commits_at_timestamps = min_timestamps.apply(get_commit_at_timestamp)
    return pd.merge(revs_and_timestamps, commits_at_timestamps.to_frame('correct_git_rev'), left_on='study', right_index=True, how='left')


def get_log_analysis_data(participant, git_rev_corrections):
    participant_level_data_raw = []
    log_analyses = get_log_analysis(participant, git_rev=git_rev_corrections.get(participant))
    conditions = log_analyses['conditions']
    base_datum = dict(participant_id=participant,
                 conditions=','.join(conditions),
                 config=log_analyses['config'],
                 git_rev=log_analyses['git_rev'])
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



# Correction task:
# Step 1: run get_correction_task.to_csv('correction_task.csv', index=False)
# Step 2: load that into Excel -> Copy to Word -> correct all typos and obvious misspellings
# Step 2.5: replace newlines with spaces, remove double-spaces. Copy back into Excel.
# Step 3: Save the result as all_writings_corrected.xlsx.

def get_correction_task(all_data):
    return all_data.query('kind == "final"').loc[:, 'final_text'].drop_duplicates().reset_index()


@mem.cache
def get_correction_task_results():
    with_corrections = pd.read_excel(str(paths.parent / 'all_writings_corrected.xlsx')).rename(columns={'finalText': 'final_text', 'corrected': 'corrected_text'})
    # Fix smartquotes.
    with_corrections['corrected_text'] = with_corrections.corrected_text.apply(lambda s: s.replace('\u2019', "'").lower())
    return with_corrections


MIN_WORD_COUNT = 5
def analyze_llks(doc, min_word_count=MIN_WORD_COUNT):
    if not isinstance(doc, str):
        return
    toks = tokenization.tokenize(doc.lower())[0]
    filtered = []
    freqs = []
    for tok in toks:
        if tok[0] not in string.ascii_letters:
            continue
        vocab_idx = analyzer.word2idx.get(tok)
        if vocab_idx is None or analyzer.counts[vocab_idx] < MIN_WORD_COUNT:
            print("Skipping", tok)
            continue
        filtered.append(tok)
        freqs.append(analyzer.log_freqs[vocab_idx])
    return pd.Series(dict(unigram_llk_mean=np.mean(freqs), unigram_llk_std=np.std(freqs), num_sentences=len(nltk.sent_tokenize(doc))))



#%%
def get_annotation_task(all_data):
    by_sentence = []
    for (participant_id, config, condition, block), text in all_data.sample(frac=1.0).set_index(['participant_id', 'config', 'condition', 'block']).corrected_text.items():
        by_sentence.append((participant_id, config, condition, block, -1, text))
        for sent_idx, sentence in enumerate(nltk.sent_tokenize(text)):
            by_sentence.append((participant_id, config, condition, block, sent_idx, sentence))
    return pd.DataFrame(by_sentence, columns=['participant_id', 'config', 'condition', 'block', 'sent_idx', 'sentence'])

# Annotation task:
# Step 1: run get_annotation_task(all_data).to_csv('by_sentence_to_annotate.csv', index=False)
# Step 2: spend a long time annotating
# Step 3: store as... some CSV file.


def get_all_annotation_results():
    return pd.read_csv('/Users/kcarnold/Downloads/by_sentence_to_annotate_allsentiment.csv')

#%%
@mem.cache
def get_latencies(participants):
    suggestion_data_raw = {participant: get_existing_requests(paths.parent / 'logs' / f'{participant}.jsonl') for participant in participants}
    suggestion_data = pd.concat({participant: pd.DataFrame(suggestions) for participant, suggestions, in suggestion_data_raw.items()}, axis=0, names=['participant_id', None])
    return suggestion_data.groupby(level=0).latency.apply(lambda x: np.percentile(x, 75)).to_frame('latency_75')
#%%
def unclean_merge_columns(df):
    return [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]

def clean_merge(*a, **kw):
    res = pd.merge(*a, **kw)
    unclean = unclean_merge_columns(res)
    assert len(unclean) == 0, unclean
    return res


def get_all_data():
    participants_by_study = get_participants_by_study()
    #[participants_by_study.study == 'sent4_0']
    participants = list(participants_by_study.participant_id)

    survey_data = {'participant': None, 'trial': None}
    raw_survey_data = get_survey_data_raw()
    def pop_and_concat(names):
        return pd.concat([raw_survey_data.pop(name) for name in names], axis=0)
    raw_survey_data['intro'] = pop_and_concat(['intro', 'intro2'])
    raw_survey_data['postTask'] = pop_and_concat(['postTask', 'postTask3'])
    raw_survey_data['postExp'] = pop_and_concat(['postExp', 'postExp3', 'postExp4'])

    for survey_name, survey_data_raw in raw_survey_data.items():
        processed_survey_data = process_survey_data(survey_name, survey_data_raw)
        is_participant_level = 'block' not in processed_survey_data.columns
        subname = 'participant' if is_participant_level else 'trial'
        merge_on = ['participant_id']
        if not is_participant_level:
            merge_on = merge_on + ['block']
        if survey_data[subname] is None:
            survey_data[subname] = processed_survey_data
        else:
            survey_data[subname] = clean_merge(
                    survey_data[subname], processed_survey_data,
                    left_on=merge_on, right_on=merge_on, how='outer')

    participant_level_data = clean_merge(
            survey_data['participant'], participants_by_study,
            left_on='participant_id', right_on='participant_id', how='outer').set_index('participant_id')

    correct_git_revs = get_correct_git_revs().set_index('participant_id').correct_git_rev.to_dict()

    log_analysis_data_raw = {participant: get_log_analysis_data(participant, correct_git_revs)
        for participant in participants}
    log_analysis_data = pd.concat({participant: pd.DataFrame(data) for participant, data in log_analysis_data_raw.items()}).reset_index(drop=True)

    trial_level_data = clean_merge(
            survey_data['trial'],
            log_analysis_data,
            left_on=['participant_id', 'block'], right_on=['participant_id', 'block'], how='outer')

    assert not unclean_merge_columns(trial_level_data)
#    trial_level_data['final_len]

    # Manual text corrections
    trial_level_data = clean_merge(
        trial_level_data, get_correction_task_results(),
        left_on='final_text', right_on='final_text', how='left')


    # Compute likelihoods
    trial_level_data = clean_merge(
        trial_level_data,
        trial_level_data.corrected_text.dropna().apply(analyze_llks),
        left_index=True, right_index=True, how='left')

    # Pull in annotations.
    annotation_results = get_all_annotation_results().query('sent_idx >= 0')#.drop('sent_idx sentence'.split(), axis=1)
    annotation_results['pos'] = pd.to_numeric(annotation_results['pos'])
    max_sentiments = annotation_results.groupby(['config', 'participant_id', 'block', 'condition']).max().loc[:,['pos', 'neg']]
    total_sentiments = annotation_results.groupby(['config', 'participant_id', 'block', 'condition']).sum().loc[:,['pos', 'neg']]
    sentiments = clean_merge(
        max_sentiments.rename(columns={'pos': 'max_positive', 'neg': 'max_negative'}),
        total_sentiments.rename(columns={'pos': 'total_positive', 'neg': 'total_negative'}),
        left_index=True, right_index=True)
    trial_level_data = clean_merge(
            trial_level_data, sentiments.reset_index(),
            left_on=['participant_id', 'config', 'block', 'condition'], right_on=['participant_id', 'config', 'block', 'condition'], how='left')

    topic_diversity = (
            annotation_results
            .dropna(subset=['topics'])
            .groupby(['participant_id', 'block', 'condition'])
            .topics
            .agg(lambda group:
                len(sorted({x.replace('-', '') for tlist in group if isinstance(tlist, str) for x in tlist.split() if x != 'nonsense'}))))
    trial_level_data = clean_merge(
        trial_level_data, topic_diversity.to_frame('num_topics').reset_index(),
        left_on=['participant_id', 'block', 'condition'], right_on=['participant_id', 'block', 'condition'], how='left')


    trial_level_data['mean_sentiment_diversity'] = (trial_level_data.total_positive + trial_level_data.total_negative - np.abs(trial_level_data.total_positive - trial_level_data.total_negative)) / trial_level_data.num_sentences

    # Calculate latency.
    participant_level_data = clean_merge(
            participant_level_data, get_latencies(participants),
            left_index=True, right_index=True, how='left')

    # Aggregate behavioral stats
    trial_level_counts = trial_level_data.loc[:, ['participant_id', 'block'] + [col for col in trial_level_data.columns if col.startswith('num_tap')]]
    taps_agg = trial_level_counts.groupby('participant_id').sum()
    participant_level_data['total_key_taps'] = taps_agg['num_tapKey']
    participant_level_data['total_rec_taps'] = taps_agg.num_tapSugg_bos + taps_agg.num_tapSugg_full + taps_agg.num_tapSugg_part
    participant_level_data['total_actions_nobackspace'] = participant_level_data['total_key_taps'] + participant_level_data['total_rec_taps']
    participant_level_data['rec_frac'] = participant_level_data['total_rec_taps'] / participant_level_data['total_actions_nobackspace']

    too_much_latency = (participant_level_data['latency_75'] > 500)
    print(f"Excluding {np.sum(too_much_latency)} for too much latency")
    too_few_actions = (
    #        (by_participant['total_sugg'] < 5) | (by_participant['num_tapKey'] < 5) |
        (participant_level_data.rec_frac < .01) | (participant_level_data.rec_frac > .99)
        )
    print(f"Excluding {np.sum(too_few_actions)} for too few actions")
    exclude = too_few_actions | too_much_latency
    print(f"Excluding {np.sum(exclude)} total")
    participant_level_data = clean_merge(
            participant_level_data, exclude.to_frame('is_excluded'),
            left_index=True, right_index=True, how='left')

    full_data = clean_merge(
            participant_level_data,
            trial_level_data,
            left_index=True, right_on='participant_id', how='right').reset_index()

#    assert participant_level_data.participant_id.value_counts().max() == 1

    omit_cols_prefixes = ['How much did you say about each topic', 'Roughly how many lines of each', 'brainstorm']
    drop_cols = [col for col in full_data.columns if any(col.startswith(x) for x in omit_cols_prefixes)]
    full_data = full_data.drop(drop_cols, axis=1)


    desired_cols = set(STUDY_COLUMNS + PARTICIPANT_LEVEL_COLUMNS + TRIAL_COLUMNS + ANALYSIS_COLUMNS + VALIDATION_COLUMNS)
    missing_cols = sorted(desired_cols - set(full_data.columns))
    extra_cols = sorted(set(full_data.columns) - desired_cols)
    print(f"Missing {len(missing_cols)} cols", missing_cols)
    print(f"{len(extra_cols)} extra cols", extra_cols)
    return full_data

all_data = get_all_data()
if False:
    all_data.to_csv('all_data.csv', index=False)
##%%
#all_data.drop(['git_rev'],axis=1).to_csv('all_data_post_fix.csv',index=False)
##%%
#pd.read_csv('all_data.csv').drop(['git_rev'], axis=1).to_csv('all_data_earlier_without_git_rev.csv',index=False)
##%%
#old_data = pd.read_csv('all_data.csv')
#merged = pd.merge(old_data, all_data, how='outer', on=['participant_id', 'block'])
##%%
#for col in merged.columns:
#    if not col.endswith('_x'):
#        continue
#    col = col[:-2]
#    this_data = merged[col+"_x"]
#    other_data = merged[col+"_y"]
#    if not np.all(this_data.values == other_data.values):
#        print("Mismatch", col)
#        assert False
##%%
#old_text = old_data.set_index(['participant_id', 'block']).final_text.fillna("MISSING")
#new_text = all_data.set_index(['participant_id', 'block']).final_text.fillna("MISSING")
#pd.merge(old_data, old_text[old_text != new_text].to_frame('equal'), left_on=['participant_id', 'block'], right_index=True).study
##%%

#%%

def summarize_freetexts(all_data):
    stacked = all_data.stack().to_frame('value')
    text_lens = stacked.groupby(level=-1).value.apply(lambda x: x.str.len().max())
    answers_to_summarize = text_lens[text_lens > 20].index.tolist()
    return all_data.loc[:, ['participant_id', 'block'] + answers_to_summarize]
#for (participant_id, block), data in summarize_freetexts(all_data[all_data.config == 'sent4']).groupby(['participant_id','block']):
#    print(participant_id, block)
#    print(data.to_dict(orient='records'))
#by_question = stacked.reset_index(level=-1).rename(columns={'level_1': 'question'})
#relevant
#%%
#    for (participant_id, block, value), other_data in
#    for pid, data_to_summarize in pd.merge(all_survey_data, answers_to_summarize.to_frame('mrl').reset_index(), left_on=['survey', 'name'], right_on=['survey', 'name']).groupby('participant_id'):
#        print(pid)
#        for (survey, idx), this_survey_data in data_to_summarize.groupby(['survey', 'idx']):
#            if survey == 'intro':
#                continue
#            print(survey, idx)
#            for row in this_survey_data.itertuples():
#                print(row.condition, row.name, '===', row.value)
#            print()
#        print()


#%%
#%%
#%%

#%%
#ambiguous = revs_and_timestamps.groupby(['participant_id', 'git_rev']).size().groupby(level=0).size() > 1