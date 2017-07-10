import os
import pandas as pd
import numpy as np
import re
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


from suggestion.analyzers import WordFreqAnalyzer, analyze_readability_measures
analyzer = WordFreqAnalyzer.build()

get_existing_reqs_cached = mem.cache(get_existing_requests)
analyze_readability_measures_cached = mem.cache(analyze_readability_measures)


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
attention_check_frac_passed
Neuroticism
NeedForCognition
OpennessToExperience
Creativity
Extraversion
Trust
LocusOfControl
CognitiveReflection
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
sugg_sentiment_mean
sugg_sentiment_std
sugg_sentiment_group_std_mean
sugg_contextual_llk_mean
'''.strip().split()

###
### Metadata
###
def get_participants_by_study():
    import yaml
    participants_table = []
    for study_name, participants in yaml.load(open(root_path / 'participants.yaml')).items():
        for participant in participants.split():
            participants_table.append((participant, study_name))
    return pd.DataFrame(participants_table, columns=['participant_id', 'study']).drop_duplicates(subset=['participant_id'])



###
### Survey data
###
def get_survey_data_raw():
    # TODO: use the prewrites too?
    # .iloc[1:] is to skip the ImportID row.
    return {name: pd.read_csv(
        os.path.join(root_path, 'surveys', name+'_responses.csv'),
        header=1, parse_dates=['StartDate', 'EndDate']).iloc[1:]
        for name in ALL_SURVEY_NAMES}

#%%
traits_key = pd.read_csv('traits.csv').set_index('item').key.reset_index().dropna().set_index('item').key.to_dict()
trait_names = {
        "N": "Neuroticism",
        "NfC": "NeedForCognition",
        "OtE": "OpennessToExperience",
        "C": "Creativity",
        "E": "Extraversion",
        "T": "Trust",
        "LoC": "LocusOfControl"}
#from collections import Counter
#sorted(Counter(v for trait in traits_key.key for v in trait.split(',')).most_common())
#%%
cogstyle_answers = {
 'A bat and a ball cost $1.10 in total. The bat costs a dollar more than the ball. How much does th...': ['5', '05', '.05', '0.05'],
 'A man buys a pig for $60, sells it for $70, buys it back for $80, and sells it finally for $90. H...': ['20'],
 'If John can drink one barrel of water in 6 days, and Mary can drink one barrel of water in 12 day...': ['4'],
 'If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 1...': ['5'],
 'In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 da...': ['47'],
 'Jerry received both the 15th highest and the 15th lowest mark in the class. How many students are...': ['29'],
 'Simon decided to invest $8,000 in the stock market one day early in 2008. Six months after he inv...': ['has lost money']}
#%%
def decode_traits(data):
    data = data.copy()
    for item, key in traits_key.items():
        datum = (data.pop(f'pers-{item}') - 3) / 4
        for trait in key.split(','):
            val = {'+': 1, '-': -1}[trait[-1]]
            name = trait_names[trait[:-1]]
            if name in data:
                data[name] += datum * val
            else:
                data[name] = datum.copy() * val

    if any(it in data.columns for it in cogstyle_answers.keys()):
        data['CognitiveReflection'] = sum([data.pop(it).isin(correct) for it, correct in cogstyle_answers.items()])
    return data

#%%

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

    if survey in ['intro']:
        data = decode_traits(data)
    return data




def get_survey_data_processed():
    survey_data = {'participant': None, 'trial': None}
    raw_survey_data = get_survey_data_raw()
    def get_and_concat(names):
        return pd.concat([raw_survey_data[name] for name in names], axis=0, ignore_index=True)
    intro_data = process_survey_data('intro', get_and_concat(['intro', 'intro2']))
    postTask_data = process_survey_data('postTask', get_and_concat(['postTask', 'postTask3']))
    postExp_data = process_survey_data('postExp', get_and_concat(['postExp', 'postExp3', 'postExp4']))

    survey_data['trial'] = postTask_data
    survey_data['participant'] = clean_merge(
            intro_data, postExp_data, on=['participant_id'], how='outer')

    return survey_data
#%%
###
### Log analysis
###

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


def get_log_analysis_data(participant, git_rev):
    data = []
    log_analysis = get_log_analysis(participant, git_rev=git_rev)
    conditions = log_analysis['conditions']
    base_datum = dict(participant_id=participant,
                 conditions=','.join(conditions),
                 config=log_analysis['config'],
                 git_rev=log_analysis['git_rev'])
    all_page_conditions = [log_analysis['byExpPage'][page]['condition'] for page in log_analysis['pageSeq']]
    for page, page_data in log_analysis['byExpPage'].items():
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
        datum.update(flatten_dict(log_analysis['blocks'][num]))
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
        data.append(datum)
    return data, all_page_conditions


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


def get_annotations_task(trial_level_data):
    by_sentence = []
    for (participant_id, config, condition, block), text in trial_level_data.sample(frac=1.0).set_index(['participant_id', 'config', 'condition', 'block']).corrected_text.dropna().items():
        by_sentence.append((participant_id, config, condition, block, -1, text))
        for sent_idx, sentence in enumerate(nltk.sent_tokenize(text)):
            by_sentence.append((participant_id, config, condition, block, sent_idx, sentence))
    res = pd.DataFrame(by_sentence, columns=['participant_id', 'config', 'condition', 'block', 'sent_idx', 'sentence'])
    res['pos'] = None
    res['neg'] = None
    res['topics'] = None
    return res

#%%
def get_sentiment_and_topic_annotations(trial_level_data, annotator):
    task = get_annotations_task(trial_level_data).set_index(['participant_id', 'config', 'condition', 'block', 'sent_idx'])

    result_files = list(paths.parent.joinpath('gruntwork').glob(f"annotations_{annotator}_*.csv"))
    if result_files:
        annotation_results = pd.concat([pd.read_csv(str(f)) for f in result_files], axis=0, ignore_index=True).dropna(how='all', subset=['pos', 'neg', 'topics'])
        task = task.combine_first(annotation_results.set_index(['participant_id', 'config', 'condition', 'block', 'sent_idx']))

    task = task.reset_index()
    topics_todo = task[task.sent_idx >= 0].groupby(['participant_id', 'block']).topics.apply(lambda group: np.all(group.isnull()))
    sentiment_todo = task[task.sent_idx >= 0].groupby(['participant_id', 'block']).pos.apply(lambda group: np.all(group.isnull()))
    todo = clean_merge(task, (topics_todo | sentiment_todo).to_frame('todo'), left_on=['participant_id', 'block'], right_index=True, how='left')
    todo = todo[todo.todo].drop(['todo'], axis=1)

    annos = task.query('sent_idx >= 0').copy()
    annos['pos'] = pd.to_numeric(annos['pos'])
    return annos, todo
#%%




MIN_WORD_COUNT = 5
def analyze_llks(doc, min_word_count=MIN_WORD_COUNT):
    if not isinstance(doc, str):
        return
    toks = tokenization.tokenize(doc.lower())[0]
    filtered = []
    freqs = []
    skipped = set()
    for tok in toks:
        if tok[0] not in string.ascii_letters:
            continue
        vocab_idx = analyzer.word2idx.get(tok)
        if vocab_idx is None or analyzer.counts[vocab_idx] < MIN_WORD_COUNT:
            skipped.add(tok)
            continue
        filtered.append(tok)
        freqs.append(analyzer.log_freqs[vocab_idx])
    if skipped:
        print("Skipped tokens:", ' '.join(sorted(skipped)))
    return pd.Series(dict(unigram_llk_mean=np.mean(freqs), unigram_llk_std=np.std(freqs), num_sentences=len(nltk.sent_tokenize(doc))))


def get_latencies(participants):
    suggestion_data_raw = {participant: get_existing_reqs_cached(paths.parent / 'logs' / f'{participant}.jsonl') for participant in participants}
    suggestion_data = pd.concat({participant: pd.DataFrame(suggestions) for participant, suggestions, in suggestion_data_raw.items()}, axis=0, names=['participant_id', None])
    return suggestion_data.groupby(level=0).latency.apply(lambda x: np.percentile(x, 75)).to_frame('latency_75')


def clean_merge(*a, must_match=[], **kw):
    res = pd.merge(*a, **kw)
    for col in must_match:
        assert res[f'{col}_x'].equals(res[f'{col}_y']), f"{col} doesn't match"
        res[col] = res.pop(f'{col}_x')
        del res[f'{col}_y']
    unclean = [col for col in res.columns if col.endswith('_x') or col.endswith('_y')]
    assert len(unclean) == 0, unclean
    assert 'index' not in res
    return res

#%%
def name_condition(meta, config):
    if meta['domain'] == 'sotu':
        return 'sotu'
    if meta['sentiment'] == 1:
        return 'sentneg'
    if meta['sentiment'] == 5:
        return 'sentpos'
    assert meta
#%%
@mem.cache
def get_suggestion_content_stats(participant_id, page_conditions):
    from suggestion import suggestion_generator
    suggestion_data_raw = get_existing_reqs_cached(paths.parent / 'logs' / f'{participant_id}.jsonl')
    prev_request_id = None
    res = []
    cur_block = []
    for sugg in suggestion_data_raw:
        meta = {k: sugg.pop(k, None) for k in 'domain sentiment_method sentiment temperature useSufarr use_bos_suggs'.split()}
        request_id = sugg['request_id']
        if prev_request_id is not None:
            if request_id == prev_request_id:
                # skip duplicate
                continue
            if request_id == 0:
                # Finished a block.
                res.append(cur_block)
                cur_block = []
            elif abs(request_id - prev_request_id) > 10:
                print(f"Warning: big skip in request ids for {participant_id}: {prev_request_id} to {request_id}")
        prev_request_id = request_id
        if not meta['domain'].startswith('yelp'):
            continue

        model = suggestion_generator.get_or_load_model(meta['domain'])
        try:
            toks = suggestion_generator.tokenize_sofar(sugg['sofar'])
        except:
            continue
        cur_word = sugg['cur_word']
        if cur_word:
            # Skip partial words.
            continue
        state = model.get_state(toks[-10:])[0]
        clf_startstate = suggestion_generator.sentiment_classifier.get_state(toks[-10:])
        phrases = [sugg[f'p{idx+1}'].split() for idx in range(3)]
        for sugg_slot, phrase in enumerate(phrases):
            if phrase:
                sentiment_posteriors = suggestion_generator.sentiment_classifier.classify_seq_by_tok(clf_startstate, phrase)
                sentiment = np.mean(sentiment_posteriors, axis=0) @ suggestion_generator.sentiment_classifier.sentiment_weights
            else:
                sentiment = None
            analyzer_indices = [analyzer.word2idx.get(tok) for tok in phrase]
            cur_block.append(dict(
                request_id=sugg['request_id'],
                sugg_slot=sugg_slot,
                sugg_contextual_llk=model.score_seq(state, phrase)[0],
                sugg_unigram_llk=np.nanmean(np.array([analyzer.log_freqs[idx] if idx is not None else np.nan for idx in analyzer_indices])),
                sugg_sentiment=sentiment))

    res.append(cur_block)


    if len(res) != len(page_conditions):
        print(f"Failed to get content stats for logfile {participant_id}")
        return {}

    by_trial = []
    for condition, block in zip(page_conditions, res):
        if condition in ['sotu', 'tweeterinchief', 'trump']:
            continue
        assert len(block) > 0
        block_df = pd.DataFrame(block)
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
#get_suggestion_content_stats('55a1db', ['sotu', 'sentpos', 'sentneg', 'sentpos', 'sentneg'])
#get_suggestion_content_stats('81519e', ['phrase', 'phrase', 'phrase', 'rarePhrase', 'rarePhrase', 'rarePhrase'])
#%%

#%%
def get_all_data_pre_annotation():
    participants_by_study = get_participants_by_study()
    #[participants_by_study.study == 'sent4_0']
    participants = list(participants_by_study.participant_id)

    survey_data = get_survey_data_processed()

    participant_level_data = clean_merge(
            survey_data['participant'].drop_duplicates('participant_id'), participants_by_study,
            on='participant_id', how='outer').set_index('participant_id')

    correct_git_revs = get_correct_git_revs().set_index('participant_id').correct_git_rev.to_dict()

    log_analysis_data_raw = {participant: get_log_analysis_data(participant, correct_git_revs.get(participant))
        for participant in participants}
    log_analysis_data = pd.concat({participant: pd.DataFrame(data) for participant, (data, page_conditions) in log_analysis_data_raw.items()}).reset_index(drop=True)

    trial_level_data = clean_merge(
            survey_data['trial'],
            log_analysis_data,
            left_on=['participant_id', 'block'], right_on=['participant_id', 'block'], how='outer')

    trial_level_data['final_length_chars'] = trial_level_data.final_text.str.len()

    # Calculate latency.
    participant_level_data = clean_merge(
            participant_level_data, get_latencies(participants),
            left_index=True, right_index=True, how='left')

    # Get suggestion content stats by trial.
    content_stats = pd.concat({
            participant_id: pd.DataFrame(get_suggestion_content_stats(participant_id, page_conditions))
            for participant_id, (data, page_conditions) in log_analysis_data_raw.items()}, names=['participant_id', 'block']).reset_index()
    trial_level_data = clean_merge(
            trial_level_data, content_stats, on=['participant_id', 'block', 'condition'], how='outer')#, must_match=['condition'])

    trial_level_data['num_taps_on_recs'] = trial_level_data.num_tapSugg_bos + trial_level_data.num_tapSugg_full + trial_level_data.num_tapSugg_part
    trial_level_data['num_nonbackspace_actions'] = trial_level_data.num_taps_on_recs + trial_level_data.num_tapKey
    trial_level_data['rec_frac'] = trial_level_data.num_taps_on_recs / trial_level_data.num_nonbackspace_actions

    # Aggregate behavioral stats
    trial_level_counts = trial_level_data.loc[:, ['participant_id', 'block'] + [col for col in trial_level_data.columns if col.startswith('num_tap') or col.startswith('attentionCheckStats')]]
    taps_agg = trial_level_counts.groupby('participant_id').sum()
    participant_level_data['total_key_taps'] = taps_agg['num_tapKey']
    participant_level_data['total_rec_taps'] = taps_agg.num_tapSugg_bos + taps_agg.num_tapSugg_full + taps_agg.num_tapSugg_part
    participant_level_data['total_actions_nobackspace'] = participant_level_data['total_key_taps'] + participant_level_data['total_rec_taps']
    participant_level_data['rec_frac_overall'] = participant_level_data['total_rec_taps'] / participant_level_data['total_actions_nobackspace']
    participant_level_data['attention_check_frac_passed_overall'] = taps_agg.pop('attentionCheckStats_passed') / taps_agg.pop('attentionCheckStats_total')

    too_much_latency = (participant_level_data['latency_75'] > 500)
    print(f"Excluding {np.sum(too_much_latency)} for too much latency")
    too_few_actions = (
    #        (by_participant['total_sugg'] < 5) | (by_participant['num_tapKey'] < 5) |
        (participant_level_data.rec_frac_overall < .01) | (participant_level_data.rec_frac_overall > .99)
        )
    print(f"Excluding {np.sum(too_few_actions)} for too few actions")
    exclude = too_few_actions | too_much_latency
    print(f"Excluding {np.sum(exclude)} total")
    participant_level_data = clean_merge(
            participant_level_data, exclude.to_frame('is_excluded'),
            left_index=True, right_index=True, how='left')

    trial_level_data['attention_check_frac_passed'] = trial_level_data.pop('attentionCheckStats_passed') / trial_level_data.pop('attentionCheckStats_total')

    return participant_level_data, trial_level_data


def get_all_data_with_annotations():
    participant_level_data, trial_level_data = get_all_data_pre_annotation()

    # Manual text corrections
    trial_level_data, corrections_todo = get_corrected_text(trial_level_data)

    # Compute likelihoods
    trial_level_data = clean_merge(
        trial_level_data,
        trial_level_data.corrected_text.dropna().apply(analyze_llks),
        left_index=True, right_index=True, how='left')

    # Compute other readability measures
    trial_level_data = clean_merge(
        trial_level_data,
        trial_level_data.corrected_text.dropna().apply(analyze_readability_measures_cached).rename(columns={'words': 'final_length_words'}),
        left_index=True, right_index=True, how='left')


    # Pull in annotations.
    annotation_results, annotation_todo = get_sentiment_and_topic_annotations(trial_level_data, annotator='kca')
    #.drop('sent_idx sentence'.split(), axis=1)
    max_sentiments = annotation_results.groupby(['config', 'participant_id', 'block', 'condition']).max().loc[:,['pos', 'neg']]
    total_sentiments = annotation_results.groupby(['config', 'participant_id', 'block', 'condition']).sum().loc[:,['pos', 'neg']]
    sentiments = clean_merge(
        max_sentiments.rename(columns={'pos': 'max_positive', 'neg': 'max_negative'}),
        total_sentiments.rename(columns={'pos': 'total_positive', 'neg': 'total_negative'}),
        left_index=True, right_index=True)
    trial_level_data = clean_merge(
            trial_level_data, sentiments.reset_index(),
            left_on=['participant_id', 'config', 'block', 'condition'], right_on=['participant_id', 'config', 'block', 'condition'], how='left')
    trial_level_data['mean_positive'] = trial_level_data['total_positive'] / trial_level_data['num_sentences']
    trial_level_data['mean_negative'] = trial_level_data['total_negative'] / trial_level_data['num_sentences']

    def has_any_nonsense(group):
        return any(x for tlist in group if isinstance(tlist, str) for x in tlist.split() if x == 'nonsense')
    def num_topics(group):
        return len(sorted({x.replace('-', '') for tlist in group if isinstance(tlist, str) for x in tlist.split() if x != 'nonsense'}))

    topic_data = (
            annotation_results
            .dropna(subset=['topics'])
            .groupby(['participant_id', 'block', 'condition'])
            .topics
            .agg([has_any_nonsense, num_topics]))

    trial_level_data = clean_merge(
        trial_level_data, topic_data.reset_index(),
        on=['participant_id', 'block', 'condition'], how='left')


    trial_level_data['mean_sentiment_diversity'] = (trial_level_data.total_positive + trial_level_data.total_negative - np.abs(trial_level_data.total_positive - trial_level_data.total_negative)) / trial_level_data.num_sentences

    full_data = clean_merge(
            participant_level_data.reset_index(),
            trial_level_data,
            on='participant_id', how='right')

    full_data['stars_before_rank'] = full_data.groupby('participant_id').stars_before.rank(method='average')
    full_data['stars_after_rank'] = full_data.groupby('participant_id').stars_after.rank(method='average')

#    assert participant_level_data.participant_id.value_counts().max() == 1

    omit_cols_prefixes = ['How much did you say about each topic', 'Roughly how many lines of each', 'brainstorm']
    drop_cols = [col for col in full_data.columns if any(col.startswith(x) for x in omit_cols_prefixes)]
    full_data = full_data.drop(drop_cols, axis=1)
    full_data = full_data.sort_values(['config', 'participant_id', 'block'])

    desired_cols = set(STUDY_COLUMNS + PARTICIPANT_LEVEL_COLUMNS + TRIAL_COLUMNS + ANALYSIS_COLUMNS + VALIDATION_COLUMNS)
    missing_cols = sorted(desired_cols - set(full_data.columns))
    extra_cols = sorted(set(full_data.columns) - desired_cols)
    print(f"Missing {len(missing_cols)} cols", missing_cols)
    print(f"{len(extra_cols)} extra cols", extra_cols)
    return full_data, corrections_todo, annotation_todo

#%%

#%%
def main(write_output=False):
    all_data, corrections_todo, annotations_todo = get_all_data_with_annotations()
    if write_output:
        all_data.to_csv('all_data.csv', index=False)
        corrections_todo.to_csv('gruntwork/corrections_todo.csv', index=False)
        annotations_todo.to_csv('gruntwork/annotations_todo_kca.csv', index=False)
    return all_data, corrections_todo, annotations_todo
#%%
def only_for_interactive():
    # global vars are a source of errors, so do this outside of global scope so that the linter can catch misuses.
    #%%
    all_data, corrections_todo, annotations_todo = main(write_output=True)

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

