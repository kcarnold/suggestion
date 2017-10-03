import pandas as pd
import numpy as np
import re
import json
from collections import Counter
import nltk
import argparse

from suggestion.paths import paths
root_path = paths.parent

import itertools
from suggestion.util import mem, flatten_dict
from suggestion import tokenization
import string


from suggestion.analysis_util import (
        # survey stuff
        skip_col_re, prefix_subs, decode_scales,
        # log analysis stuff
        classify_annotated_event, get_log_analysis,
        get_content_stats_single_suggestion)


from suggestion.analyzers import WordFreqAnalyzer, analyze_readability_measures
word_freq_analyzer = WordFreqAnalyzer.get()

analyze_readability_measures_cached = mem.cache(analyze_readability_measures)


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
rec_frac_overall
attention_check_frac_passed_overall
Neuroticism
NeedForCognition
OpennessToExperience
Creativity
Extraversion
Trust
LocusOfControl
CognitiveReflection
Imagination
Agreeableness
'''.strip().split()

# Can't do these two without manual history editing:
#attention_check_fullword
#attention_check_partword


TRIAL_COLUMNS = '''
block
condition
latency_75
know_what_to_write
stars_before
stars_after
stars_before_rank
stars_after_rank
stars_before_z
stars_after_z
stars_before_groupz
stars_after_groupz
self_report_accuracy
final_text
attention_check_frac_passed_trial
rec_frac_trial
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
total_pos
total_neg
max_pos
max_neg
mean_pos
mean_neg
num_topics
mtld
pairdist_words_mean
pairdist_words_std
pairdist_sentences_mean
pairdist_sentences_std
'''.strip().split()

VALIDATION_COLUMNS = '''
sugg_unigram_llk_mean
sugg_sentiment_mean
sugg_sentiment_std
sugg_sentiment_group_std_mean
sugg_contextual_llk_mean
'''.strip().split()

###
### Metadata
###
def get_participants_by_study(batches=None):
    import yaml
    participants_table = []
    for study_name, participants in yaml.load(open(root_path / 'participants.yaml')).items():
        if batches and study_name not in batches:
            continue
        for participant in participants.split():
            participants_table.append((participant, study_name))
    return pd.DataFrame(participants_table, columns=['participant_id', 'study']).drop_duplicates(subset=['participant_id'])



#%%
traits_key = pd.read_csv(paths.parent / 'traits.csv').set_index('item').key.reset_index().dropna().set_index('item').key.to_dict()
trait_names = {
        "N": "Neuroticism",
        "NfC": "NeedForCognition",
        "OtE": "OpennessToExperience",
        "C": "Creativity",
        "E": "Extraversion",
        "T": "Trust",
        "LoC": "LocusOfControl",
        "I": "Imagination",
        "A": "Agreeableness"}

#%%
trial_level_renames = {'other': 'other_comments_trial', 'techDiff': 'tech_difficulties_trial'}
trial_level_renames.update({
        x: 'tlx_'+x for x in
        ["mental", "physical", "temporal", "performance", "effort", "frustration"]})

def decode_surveys(log_analysis, active_traits):
    controlled_inputs = log_analysis['allControlledInputs']
    personality = {}
    by_trial = [{} for condition in log_analysis['conditions']]
    other = {}
    for k, v in controlled_inputs.items():
        if k.startswith("restaurant"):
            continue
        assert '-' in k
        part, item = k.rsplit('-', 1)
        if item in traits_key or item[:-1] in traits_key:
            if item not in traits_key:
                assert item[-1] == '.'
                item = item[:-1]
            for trait in traits_key[item].split(','):
                val = {'+': 1, '-': -1}[trait[-1]]
                name = trait[:-1]
                if name in active_traits:
                    personality.setdefault(name, []).append((item, val * v))
        elif part.startswith('postTask'):
            trial_num = int(part.rsplit('-', 1)[1])
            item = trial_level_renames.get(item, item)
            by_trial[trial_num][item] = v
        else:
            assert k.startswith('postExp')
            k = k.split('-', 1)[1]
            other[k] = v
    return personality, other, by_trial

#%%
def aggregate_personality(personality):
    return {trait_names[trait]: np.mean([val for item, val in items]) for trait, items in personality.items()}
#%%

def summarize_trials(log_analysis):
    data = []
    conditions = log_analysis['conditions']
    participant_id = log_analysis['participant_id']
    base_datum = dict(participant_id=participant_id,
                 conditions=','.join(conditions),
                 config=log_analysis['config'],
                 git_rev=log_analysis['git_rev'])
    for page in log_analysis['pageSeq']:
        page_data = log_analysis['byExpPage'][page].copy()
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

        # Count actions.
        actions = page_data.pop('actions')
        classified_events = [classify_annotated_event(evt) for evt in actions]
        for typ, count in Counter(classified_events).items():
            if typ is not None:
                datum[f'num_{typ}'] = count

        # Summarize latencies
        # Note that 'displayedSuggs' is actually indexed by context sequence number,
        # and some contexts never have a corresponding displayed suggestion
        # if the latency is too high.
        displayedSuggs = page_data.pop('displayedSuggs')
        page_data.pop('chunks')
        words = page_data.pop('words')

        datum.update(flatten_dict(page_data))

        if len(displayedSuggs):
            latencies = [rec['latency'] for rec in displayedSuggs if rec]
            if latencies:
                datum['latency_75_trial'] = np.percentile(latencies, 75)
            datum['num_displayed_suggs'] = len(latencies)
            datum['num_missing_suggs'] = len([rec for rec in displayedSuggs if not rec])
            datum['frac_missing_suggs'] = datum['num_missing_suggs'] / len(displayedSuggs)


            # Compute efficiencies.
            num_inserted_full = num_could_have_inserted_full = num_inserted_sugg = num_could_have_inserted_sugg = 0
            extra_chars = ',.!? \n'
            for word in words:
                could_have_matched_at = None
                did_insert = False
                word_text = ''.join(chunk['chars'] for chunk in word['chunks'])
                word_trim = word_text.strip(extra_chars)
                for chunk_idx, chunk in enumerate(word['chunks']):
                    predictions = chunk['action']['visibleSuggestions']['predictions']
                    suggested_words = [''.join(pred['words'][:1]) for pred in predictions]
                    if word_trim in suggested_words:
                        if could_have_matched_at is None:
                            could_have_matched_at = chunk_idx
                    if chunk['actionClass'].startswith('tapSugg'):
                        if chunk['action']['which'] == 'synonyms':
                            # Synonyms aren't about efficiency.
                            continue
                        stripped_inserted = chunk['action']['sugInserted'].strip(extra_chars)
                        if stripped_inserted == chunk['chars'].strip(extra_chars) and len(''.join(chunk['chars'] for chunk in word['chunks'][chunk_idx + 1:]).strip(extra_chars)) == 0:
                        # if word_trim in suggested_words and :
                            did_insert = True
                            # Things are a bit weird if the sugg inserted a punctuation
                            assert word_trim in suggested_words or len(stripped_inserted) == 0
                            break
                        else:
                            print("Inserted but backspaced...", participant_id, chunk['action']['sugInserted'], chunk['chars'].strip(extra_chars))
                if did_insert:
                    num_inserted_sugg += 1
                    if len(word['chunks']) == 1:
                        num_inserted_full += 1
                if could_have_matched_at is not None:
                    if could_have_matched_at == 0:
                        num_could_have_inserted_full += 1
                    num_could_have_inserted_sugg += 1
            datum['num_inserted_full'] = num_inserted_full
            datum['num_could_have_inserted_full'] = num_could_have_inserted_full
            datum['num_inserted_sugg'] = num_inserted_sugg
            datum['num_could_have_inserted_sugg'] = num_could_have_inserted_sugg

            datum['efficiency_full'] = num_inserted_full / num_could_have_inserted_full if num_could_have_inserted_full else np.nan
            datum['efficiency_all'] = num_inserted_sugg / num_could_have_inserted_sugg if num_could_have_inserted_sugg else np.nan
        else:
            assert page_data['condition'] == 'zerosugg'

        renames = {
            'finalText': 'final_text',
            'place_knowWhatToWrite': 'know_what_to_write',
            'place_stars': 'stars_before'}
        for old_name, new_name in renames.items():
            if old_name in datum:
                datum[new_name] = datum.pop(old_name)
        data.append(datum)
    return data


#%% Human tasks
# Correction task:
# Step 1: run corrections_todo.to_csv('corrections_todo.csv', index=False)
# Step 2: load that into Excel -> Copy to Word -> correct all typos and obvious misspellings.
# Step 3: Save the result as gruntwork/correction_batch_N.csv, IN UTF-8


def get_corrected_text(trial_level_data):
    trial_level_data['final_text_for_correction'] = trial_level_data['final_text'].str.replace(re.compile(r'\s+'), ' ')
    result_files = list(paths.parent.joinpath('gruntwork').glob("corrections_batch*.csv"))
    if result_files:
        correction_results = pd.concat([pd.read_csv(str(f)) for f in result_files], axis=0, ignore_index=True)
        assert correction_results.columns.tolist() == ['final_text', 'corrected_text']
        correction_results['final_text_for_correction'] = correction_results['final_text'].str.replace(re.compile(r'\s+'), ' ')
        correction_results['corrected_text'] = correction_results.corrected_text.apply(lambda s: s.replace('\u2019', "'").lower())
        trial_level_data = clean_merge(
            trial_level_data, correction_results.drop(['final_text'], axis=1),
            on='final_text_for_correction', how='left')
    else:
        trial_level_data['corrected_text'] = None

    corrections_todo = trial_level_data[trial_level_data.corrected_text.isnull()].final_text_for_correction.dropna().drop_duplicates().to_frame('final_text')
    corrections_todo['corrected_text'] = None

    return trial_level_data, corrections_todo

#%%
# Annotation task:
# Step 1: run get_annotation_task(all_data).to_csv('by_sentence_to_annotate2.csv', index=False)
# Step 2: spend a long time annotating
# Step 3: store as... some CSV file.


def get_sentiment_annotations_task(trial_level_data):
    by_sentence = []
    for (participant_id, config, condition, block), text in trial_level_data.sample(frac=1.0).set_index(['participant_id', 'config', 'condition', 'block']).corrected_text.dropna().items():
        by_sentence.append((participant_id, config, condition, block, -1, text))
        for sent_idx, sentence in enumerate(nltk.sent_tokenize(text)):
            by_sentence.append((participant_id, config, condition, block, sent_idx, sentence))
    res = pd.DataFrame(by_sentence, columns=['participant_id', 'config', 'condition', 'block', 'sent_idx', 'sentence'])
    return res

#%%
def merge_sentiment_annotations(task, sent_annotation_results):
    if len(sent_annotation_results):
        task = clean_merge(
            task,
            sent_annotation_results.drop(['sentence'], axis=1),
            how='left', on=['participant_id', 'config', 'condition', 'block', 'sent_idx'])#, must_match=['sentence'])
    else:
        task = task.copy()
        task['pos'] = None
        task['neg'] = None
        task['nonsense'] = None
#    topics_todo = task[task.sent_idx >= 0].groupby(['participant_id', 'block']).topics.apply(lambda group: np.all(group.isnull()))
    todo_flag = task[task.sent_idx >= 0].groupby(['participant_id', 'block']).pos.apply(lambda group: np.all(group.isnull()))
    todo = clean_merge(task, todo_flag.to_frame('todo'), left_on=['participant_id', 'block'], right_index=True, how='left')
    todo = todo[todo.todo].drop(['todo'], axis=1)

    annos = task.query('sent_idx >= 0').copy()
    for col in ['pos', 'neg', 'nonsense']:
        annos[col] = pd.to_numeric(annos[col])
    return annos, todo
#%%

MIN_WORD_COUNT = 5
def analyze_llks(doc, min_word_count=MIN_WORD_COUNT):
    try:
        toks = tokenization.tokenize(doc.lower())[0]
    except Exception:
        print("Failed to tokenize", doc[:30])
        return pd.Series()
    filtered = []
    freqs = []
    skipped = set()
    for tok in toks:
        if tok[0] not in string.ascii_letters:
            continue
        vocab_idx = word_freq_analyzer.word2idx.get(tok)
        if vocab_idx is None or word_freq_analyzer.counts[vocab_idx] < MIN_WORD_COUNT:
            skipped.add(tok)
            continue
        filtered.append(tok)
        freqs.append(word_freq_analyzer.log_freqs[vocab_idx])
    if skipped:
        print("Skipped tokens:", ' '.join(sorted(skipped)))
    return pd.Series(dict(unigram_llk_mean=np.mean(freqs), unigram_llk_std=np.std(freqs), num_sentences=len(nltk.sent_tokenize(doc))))


def clean_merge(*a, must_match=[], combine_cols=[], **kw):
    res = pd.merge(*a, **kw)
    for col in must_match:
        assert res[f'{col}_x'].equals(res[f'{col}_y']), f"{col} doesn't match"
        res[col] = res.pop(f'{col}_x')
        del res[f'{col}_y']
    for col in combine_cols:
        res[col] = res.pop(f'{col}_x').combine_first(res.pop(f'{col}_y'))
    unclean = [col for col in res.columns if col.endswith('_x') or col.endswith('_y')]
    assert len(unclean) == 0, unclean
    assert 'index' not in res
    return res

#%%

def get_log_pages(analyzed):
    res = []
    for page_name in analyzed['pageSeq']:
        page = analyzed['byExpPage'][page_name]
        res.append((page_name, page['condition'], page['displayedSuggs']))
    return res


@mem.cache
def get_suggestion_content_stats(pages):
    by_trial = []
    for page_name, condition, displayed_suggs in pages:
        if page_name.startswith('pract'):
            #condition in ['sotu', 'tweeterinchief', 'trump', 'nosugg', 'airbnb']:
            continue
        if condition == 'zerosugg':
            by_trial.append(dict())
            continue
        assert len(displayed_suggs) > 0

        block_data = []
        for sugg in displayed_suggs:
            if sugg is None:
                # Skip contexts where no suggestion was shown.
                continue

            for datum in get_content_stats_single_suggestion(sugg, word_freq_analyzer=word_freq_analyzer) or []:
                block_data.append(datum)

        assert len(block_data) > 0
        block_df = pd.DataFrame(block_data)
        assert 'request_id' in block_df
        block_df = block_df.drop_duplicates(['request_id', 'sugg_slot'])
        by_trial.append(dict(
            condition=condition,
            sugg_unigram_llk_mean=block_df.sugg_unigram_llk.mean(),
#            sugg_unigram_llk_std=block_df.sugg_unigram_llk.std(), # not clearly meaningful, since it's already a mean
            sugg_sentiment_mean=block_df.sugg_sentiment.mean(),
            sugg_sentiment_std=block_df.sugg_sentiment.std(),
            sugg_sentiment_group_std_mean=block_df.groupby('request_id').sugg_sentiment.std().mean(),
            sugg_contextual_llk_mean=block_df.sugg_contextual_llk.mean()))
    return by_trial
#%%

def drop_cols_by_prefix(df, prefixes):
    drop_cols = [col for col in df.columns if any(col.startswith(x) for x in prefixes)]
    return df.drop(drop_cols, axis=1)
#%%
def summarize_times(log_analysis):
   times = log_analysis['screenTimes']
   return dict(total_time_mins=(times[-1]['timestamp'] - times[1]['timestamp']) / 1000 / 60)

#%%
def get_all_data_pre_annotation(batches=None):
    participants_by_study = get_participants_by_study(batches=batches)
    participants = list(participants_by_study.participant_id)

    participant_level_data = []
    trial_level_data = []
    for participant in participants:
        log_analysis_data_raw = get_log_analysis(participant)
        personality_raw, other_participant_level, surveys_by_trial = decode_surveys(log_analysis_data_raw, 'N,NfC,OtE,E'.split(','))
        participant_level_data.append(dict(
                study=participants_by_study.set_index('participant_id').study.loc[participant],
                participant_id=participant,
                **other_participant_level,
                **aggregate_personality(personality_raw),
                **summarize_times(log_analysis_data_raw)))

        log_data_by_trial = summarize_trials(log_analysis_data_raw)
        suggestion_content_by_trial = get_suggestion_content_stats(get_log_pages(log_analysis_data_raw))


        for trial_id, (survey_data, log_data, sug_content_data) in enumerate(itertools.zip_longest(surveys_by_trial, log_data_by_trial, suggestion_content_by_trial)):
            assert participant == log_data.pop('participant_id')
            assert trial_id == log_data.pop('block')
#            assert participant == sug_content_data.pop('participant_id')
#            assert trial_id == sug_content_data.pop('block')
#            assert log_data['condition'] == sug_content_data.pop('condition')


            trial_level_data.append(dict(
                    participant_id=participant,
                    block=trial_id,
                    **survey_data,
                    **log_data))

    # Pandify.
    participant_level_data = pd.DataFrame(participant_level_data).set_index('participant_id')
    trial_level_data = pd.DataFrame(trial_level_data)


    # Bin traits by percentile
    for trait in trait_names.values():
        if trait in participant_level_data.columns:
            participant_level_data[trait+"_hi"] = participant_level_data.groupby('study')[trait].transform(lambda x: x > np.nanpercentile(x, 50))

    trial_level_data['final_length_chars'] = trial_level_data.final_text.str.len()

    # Fill missing data in tap counts with zeros
    for col in trial_level_data.columns:
        if col.startswith('num_tap'):
            trial_level_data[col] = trial_level_data[col].fillna(0)

    trial_level_data['num_taps_on_recs'] = trial_level_data.num_tapSugg_bos + trial_level_data.num_tapSugg_full + trial_level_data.num_tapSugg_part
    trial_level_data['num_nonbackspace_actions'] = trial_level_data.num_taps_on_recs + trial_level_data.num_tapKey
    trial_level_data['rec_frac_trial'] = trial_level_data.num_taps_on_recs / trial_level_data.num_nonbackspace_actions

    # Aggregate behavioral stats
    trial_level_counts = trial_level_data.loc[:, ['participant_id', 'block'] + [col for col in trial_level_data.columns if col.startswith('num_tap') or col.startswith('attentionCheckStats')]]
    taps_agg = trial_level_counts.groupby('participant_id').sum()
    participant_level_data['total_key_taps_overall'] = taps_agg['num_tapKey']
    participant_level_data['total_rec_taps_overall'] = taps_agg.num_tapSugg_bos + taps_agg.num_tapSugg_full + taps_agg.num_tapSugg_part
    participant_level_data['total_actions_nobackspace_overall'] = participant_level_data['total_key_taps_overall'] + participant_level_data['total_rec_taps_overall']
    participant_level_data['rec_frac_overall'] = participant_level_data['total_rec_taps_overall'] / participant_level_data['total_actions_nobackspace_overall']
    if 'attentionCheckStats_passed' in taps_agg:
        participant_level_data['attention_check_frac_passed_overall'] = taps_agg.pop('attentionCheckStats_passed') / taps_agg.pop('attentionCheckStats_total')

    for col in ['rec_frac_trial', 'efficiency_all', 'efficiency_full', 'latency_75_trial']:
        participant_level_data[f'{col}_max'] = trial_level_data.groupby('participant_id')[col].max()
        participant_level_data[f'{col}_min'] = trial_level_data.groupby('participant_id')[col].min()

    participant_level_data = participant_level_data.reset_index()


    # Filter for valid data.
    trial_level_data = trial_level_data[~trial_level_data.final_text.isnull()]
    participant_level_data = participant_level_data[participant_level_data.participant_id.isin(trial_level_data[~trial_level_data.final_text.isnull()].participant_id.unique())]

    trial_level_data['argue_pro'] = trial_level_data['place_idx'] < 2

    # For a summary of how many trials there are each:
    # trial_level_data.groupby(['config', 'participant_id']).size().groupby(level=0).value_counts()

    return tuple(
            table.dropna(axis=1, how='all') for table in
            (participant_level_data, trial_level_data))


def get_all_data_with_annotations(batches=None):
    participant_level_data, trial_level_data = get_all_data_pre_annotation(batches=batches)

    # Manual text corrections
    trial_level_data, corrections_todo = get_corrected_text(trial_level_data)
    # Fill in missing.
    trial_level_data['corrected_text'] = trial_level_data.corrected_text.combine_first(trial_level_data['final_text'])

    # Compute likelihoods
    corrected_text = trial_level_data.corrected_text.dropna()
    trial_level_data = clean_merge(
        trial_level_data,
        corrected_text.apply(analyze_llks),
        left_index=True, right_index=True, how='left')

    # Compute other readability measures
    trial_level_data = clean_merge(
        trial_level_data,
        trial_level_data.corrected_text.dropna().apply(analyze_readability_measures_cached).rename(columns={'words': 'final_length_words'}),
        left_index=True, right_index=True, how='left')


    # Pull in annotations.
    sentiment_annotations_task = get_sentiment_annotations_task(trial_level_data)
    turk_sentiment_annotations_results = load_turk_sentiment_annotations()
    sent_annotation_results, sent_annotation_todo = merge_sentiment_annotations(sentiment_annotations_task, aggregate_turk_sentiment_annotations(turk_sentiment_annotations_results))
    #.drop('sent_idx sentence'.split(), axis=1)
    sent_annotation_results['is_mixed'] = (sent_annotation_results['pos'] > 0) & (sent_annotation_results['neg'] > 0)
    sent_annotation_results['intensity'] = sent_annotation_results['pos'] + sent_annotation_results['neg']
    sent_annotation_results['polarity'] = sent_annotation_results['pos'] - sent_annotation_results['neg']
    max_sentiments = sent_annotation_results.groupby(['config', 'participant_id', 'block', 'condition']).max().loc[:,['pos', 'neg', 'nonsense', 'intensity', 'polarity']]
    total_sentiments = sent_annotation_results.groupby(['config', 'participant_id', 'block', 'condition']).sum().loc[:,['pos', 'neg', 'intensity', 'polarity']]


    def prefix_columns(df, prefix, cols):
        return df.rename(columns={col: f'{prefix}{col}' for col in cols})

    sentiments = clean_merge(
        prefix_columns(max_sentiments, 'max_', 'pos neg intensity polarity nonsense'.split()),
        prefix_columns(total_sentiments, 'total_', 'pos neg intensity polarity'.split()),
        left_index=True, right_index=True)
    sentiments = clean_merge(
        sentiments,
        sent_annotation_results.groupby(['config', 'participant_id', 'block', 'condition']).is_mixed.mean().to_frame('mean_sentiment_mixture'),
        left_index=True, right_index=True)
    trial_level_data = clean_merge(
            trial_level_data, sentiments.reset_index(),
            left_on=['participant_id', 'config', 'block', 'condition'], right_on=['participant_id', 'config', 'block', 'condition'], how='left')
    for col in 'pos neg intensity polarity'.split():
        trial_level_data[f'mean_{col}'] = trial_level_data[f'total_{col}'] / trial_level_data['num_sentences']
    trial_level_data['any_positive'] = trial_level_data['max_pos'] >= 1
    trial_level_data['any_negative'] = trial_level_data['max_neg'] >= 1

    trial_level_data['has_any_nonsense'] = trial_level_data['max_nonsense'] >= 0.5
    participant_level_data = clean_merge(
            participant_level_data,
            trial_level_data.groupby('participant_id').has_any_nonsense.max().to_frame('participant_wrote_any_nonsense'),
            left_on='participant_id', right_index=True, how='left')

#    persuasiveness_annos_task = get_persuasiveness_annotations(trial_level_data)
    # TODO: merge in results.


    # trial_level_data['mean_sentiment_diversity'] = (trial_level_data.total_positive + trial_level_data.total_negative - np.abs(trial_level_data.total_positive - trial_level_data.total_negative)) / trial_level_data.num_sentences

#    trial_level_data['stars_before_rank'] = trial_level_data.groupby('participant_id').stars_before.rank(method='average')
#    trial_level_data['stars_after_rank'] = trial_level_data.groupby('participant_id').stars_after.rank(method='average')

    omit_cols_prefixes = ['How much did you say about each topic', 'Roughly how many lines of each', 'brainstorm']

    participant_level_data = drop_cols_by_prefix(participant_level_data, omit_cols_prefixes)
    trial_level_data = drop_cols_by_prefix(trial_level_data, omit_cols_prefixes)

    full_data = clean_merge(
            participant_level_data,
            trial_level_data,
            on='participant_id', how='right')

    full_data = full_data.sort_values(['config', 'participant_id', 'block'])

    column_order = STUDY_COLUMNS + PARTICIPANT_LEVEL_COLUMNS + TRIAL_COLUMNS + ANALYSIS_COLUMNS + VALIDATION_COLUMNS
    full_data = reorder_columns(full_data, column_order)
    participant_level_data = reorder_columns(participant_level_data, column_order)
    trial_level_data = reorder_columns(trial_level_data, column_order)

    desired_cols = set(column_order)
    missing_cols = sorted(desired_cols - set(full_data.columns))
    extra_cols = sorted(set(full_data.columns) - desired_cols)
    print(f"Missing {len(missing_cols)} cols", missing_cols)
    print(f"{len(extra_cols)} extra cols", extra_cols)

#    annotations_plus_exclusions = clean_merge(
#            participant_level_data.set_index('participant_id').loc[:, ['is_excluded']],
#            sent_annotation_todo,
#            left_index=True, right_on='participant_id')
#    non_excluded_annotations = annotations_plus_exclusions.query('not is_excluded')
#    non_excluded_annotations = non_excluded_annotations.drop('is_excluded', axis=1)

    return dict(
            all_data=full_data,
            participant_level_data=participant_level_data,
            trial_level_data=trial_level_data,
            corrections_todo=corrections_todo,
            sent_annotation_todo=sent_annotation_todo,
#            persuasiveness_annos_todo=persuasiveness_annos_task
            )

#%%
def load_turk_sentiment_annotations(result_files=None):
    meta_header = ['participant_id', 'config', 'condition', 'block']
    if result_files is None:
        result_files = list(paths.parent.joinpath('gruntwork', 'turk_sentiment_annotations_results').glob("Batch*results.csv"))
    if not result_files:
        print("No Turk annotation results found.")
        return pd.DataFrame([], columns=['WorkerId', 'block', 'condition', 'config', 'neg', 'nonsense', 'participant_id', 'pos', 'sent_idx', 'sentence'])

    raw = pd.concat([pd.read_csv(str(f)) for f in result_files], axis=0, ignore_index=True)
    records = raw.loc[:, ['WorkerId', 'Answer.results']].to_dict('records')
    res = []
    for record in records:
        worker_id = record['WorkerId']
        annos = json.loads(record['Answer.results'])
        for text_entry in annos:
            meta = dict(dict(zip(meta_header, text_entry['meta'])), WorkerId=worker_id)
            for sent in text_entry['data']:
                res.append(dict(meta, **sent))
    res = pd.DataFrame(res)#.fillna(0)
    assert res['sentIdx'].equals(res['sent_idx'])
    del res['sentIdx']
    res['pos'] = pd.to_numeric(res['pos'])
    res['neg'] = pd.to_numeric(res['neg'])
    return res

def aggregate_turk_sentiment_annotations(annotations):
    if len(annotations) == 0:
        return pd.DataFrame()
    mean_anno = annotations.groupby(['participant_id', 'config', 'condition', 'block', 'sent_idx', 'sentence']).mean().loc[:, ['pos', 'neg', 'nonsense']]
    from nltk.metrics.agreement import AnnotationTask, binary_distance
    def get_alpha_overall(annos):
        data = []
        for idx, row in annos.iterrows():
            for col in ['pos', 'neg', 'nonsense']:
                data.append((
                    row.WorkerId,
                    '{participant_id}-{block}-{sent_idx}-{col}'.format(**row, col=col),
                    row[col]))
        return AnnotationTask(data).alpha()
    def interval_distance(a, b):
    #    return abs(a-b)
        return pow(a-b, 2)

    def get_alpha_single_col(annos, col):
        if col == 'nonsense':
            distance = binary_distance
        else:
            distance = interval_distance
        data = [
            (row.WorkerId, '{participant_id}-{block}-{sent_idx}'.format(**row), row[col])
            for idx, row in annos.iterrows()
            if not np.isnan(row[col])]
        return AnnotationTask(data, distance=distance).alpha()
    cols = ['pos', 'neg', 'nonsense']
    base_alphas = {col: get_alpha_single_col(annotations, col) for col in cols}
    # worker_ids = annotations.WorkerId.unique()
    # alpha_without = {
    #     worker_id: {
    #         col: get_alpha_single_col(annos[annos.WorkerId != worker_id], col)
    #         for col in cols}
    #     for worker_id in worker_ids}
    print(base_alphas)
    return mean_anno.reset_index()
#%%
def interval_distance(a, b):
#    return abs(a-b)
    return pow(a-b, 2)


#%%
from nltk.metrics.agreement import AnnotationTask, binary_distance

def get_alpha(annotations, annotator, item_format, result, distance_type='interval'):
    if distance_type == 'binary':
        distance = binary_distance
    elif distance_type == 'interval':
        distance = interval_distance
    else:
        raise KeyError(f"Unknown type of distance: {distance_type}")
    data = [
        (row[annotator], item_format.format(**row), row[result])
        for idx, row in annotations.iterrows()
        if not np.isnan(row[result])]
    return AnnotationTask(data, distance=distance).alpha()


def load_turk_persuasiveness_annotations(result_files=None):
    if result_files is None:
        result_files = list(paths.parent.joinpath('gruntwork', 'turk_persuasiveness_results').glob('Batch*results.csv'))
    if not result_files:
        print("No Turk persuasiveness results found.")
        return pd.DataFrame([])

    raw = pd.concat([pd.read_csv(str(f)) for f in result_files], axis=0, ignore_index=True)
    res = []
    for record in raw.loc[:, ['WorkerId', 'Answer.results']].to_dict('records'):
        worker_id = record['WorkerId']
        for entry in json.loads(record['Answer.results']):
            res.append(dict(worker_id=worker_id, **entry))
    res = pd.DataFrame(res)
    del res['Index']
    return res
p_anno = load_turk_persuasiveness_annotations()
p_anno['votes_z'] = p_anno.groupby('worker_id').votes.transform(lambda x: (x-x.mean())/np.maximum(1, x.std()))
get_alpha(p_anno, 'worker_id', '{final_text}', 'votes_z')
#%%
def reorder_columns(df, desired_order):
    reorder_cols = []
    for col in desired_order:
        if col in df.columns:
            reorder_cols.append(col)
    for col in df.columns:
        if col not in reorder_cols:
            reorder_cols.append(col)
    return df.loc[:, reorder_cols]

#%%
def get_sent_annotation_json(sent_annotation_todo):
    groups = [
        (key, group.loc[:, ['sent_idx', 'sentence']].to_dict(orient='records'))
        for key, group
        in sent_annotation_todo.groupby(['participant_id', 'config', 'condition', 'block'], sort=False)]
    import random
    random.shuffle(groups)
    return groups


#%%
def get_persuasiveness_anno_json(trial_level_data, avoid):
    rs = np.random.RandomState(0)
    subset = trial_level_data[trial_level_data.argue_pro != avoid]
    trial_iters = [(item._asdict() for item in group.loc[:, ['participant_id', 'block', 'final_text']].sample(frac=1.0, random_state=rs).itertuples()) for participant_id, group in subset.groupby('participant_id')]
    rs.shuffle(trial_iters)
    return list(
            itertools.chain.from_iterable(trial_iters))


#%%
def main(args):
    x = get_all_data_with_annotations(batches=args.batch)
    basename = '+'.join(args.batch) if len(args.batch) else 'ALL'
    if args.write_output:
        x['all_data'].to_csv(f'{basename}_all_data.csv', index=False)
        x['participant_level_data'].to_csv(f'{basename}_participant_level_data.csv', index=False)
        x['trial_level_data'].to_csv(f'{basename}_trial_level_data.csv', index=False)
        x['corrections_todo'].to_csv(f'gruntwork/{basename}_corrections_todo.csv', index=False)
        x['sent_annotation_todo'].to_csv(f'gruntwork/{basename}_annotations_todo_kca.csv', index=False)
        json.dump(get_sent_annotation_json(x['sent_annotation_todo']), open(f'gruntwork/{basename}_annotations_todo.json', 'w'),
                  default=lambda x: x.tolist())
        json.dump(get_persuasiveness_anno_json(x['trial_level_data'], avoid=False), open(f'gruntwork/{basename}_persuasive_anno_todo.json', 'w'),
                  default=lambda x: x.tolist())
        json.dump(get_persuasiveness_anno_json(x['trial_level_data'], avoid=True), open(f'gruntwork/{basename}_avoid_anno_todo.json', 'w'),
                  default=lambda x: x.tolist())
    return x
#%%

def get_arnold16_annotation_json():
    arnold16 = pd.read_csv('data/arnold16_full_participant_data.csv')
    by_sentence = []
    for (participant_id, condition, block), text in arnold16.sample(frac=1.0).set_index(['participant_id', 'condition', 'idx']).reviewText.items():
        config = 'arnold16'
        participant_id = str(participant_id)
        by_sentence.append((participant_id, config, condition, block, -1, text))
        for sent_idx, sentence in enumerate(nltk.sent_tokenize(text)):
            by_sentence.append((participant_id, config, condition, block, sent_idx, sentence))
    res = pd.DataFrame(by_sentence, columns=['participant_id', 'config', 'condition', 'block', 'sent_idx', 'sentence'])
    json.dump(get_sent_annotation_json(res), open(f'gruntwork/arnold16_annotations_todo.json', 'w'),
          default=lambda x: x.tolist())
    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-output', action='store_true')
    parser.add_argument('--batch', action='append')
    args = parser.parse_args()
    # global vars are a source of errors, so we do this convoluted thing so the linter doesn't think they're valid globals.
    globals().update(main(args))

#%%
