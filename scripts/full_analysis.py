import re
import os
import pathlib
import pickle
import subprocess
import pandas as pd
import json
import numpy as np
import datetime
import itertools

from suggestion.util import mem

from suggestion.analysis_util import get_existing_requests

batch_code = 'sent3_2'

#%%
if '__file__' in globals():
    root_path = pathlib.Path(__file__).resolve().parent.parent
else:
    root_path = pathlib.Path('~/code/suggestion').expanduser()
#%%

def get_survey_seq(batch_code):
    seq = [('intro', 0)]
    batch_code_basename = batch_code.split('_', 1)[0]
    if batch_code_basename in ['sent3', 'sent32', 'polarized']:
        for i in range(3):
            seq.append(('postTask3', i))
        seq.append(('postExp3', 0))
    else:
        for i in range(2):
            seq.append(('postTask', i))
        seq.append(('postExp', 0))
    return seq


import yaml
participants = yaml.load(open(root_path / 'participants.yaml'))[batch_code].split()
assert len(participants) == len(set(participants)), "Duplicate participants"
#%%
@mem.cache
def get_suggestion_data(participant):
    return get_existing_requests(root_path / 'logs' / f'{participant}.jsonl')

suggestion_data_raw = {participant: get_suggestion_data(participant) for participant in participants}
suggestion_data = pd.concat({participant: pd.DataFrame(suggestions) for participant, suggestions, in suggestion_data_raw.items()}, axis=0)
#%%
if False:
    #%%
    json.dump(suggestion_data_raw, open(f'{batch_code}_sugdata.json', 'w'))
#%%
#suggestion_data.to_csv(f'{batch_code}_suggestion_stats.csv')
#%%




#%%

def classify_annotated_event(evt):
    typ = evt['type']
    if typ in {'externalAction', 'next'}:
        return None
    text = evt['curText']
    null_word = len(text) == 0 or text[-1] == ' '
    text = text.strip()
    bos = len(text) == 0 or text[-1] in '.?!'
    if typ == 'tapKey':
        return 'tapKey'
    if typ == 'tapBackspace':
        return 'tapBackspace'
    if typ == 'tapSuggestion':
        if bos:
            sugg_mode = 'bos'
        elif null_word:
            sugg_mode = 'full'
        else:
            sugg_mode = 'part'
        return 'tapSugg_' + sugg_mode
    assert False, typ

from collections import Counter
#%%
def flatten_dict(x, prefix=''):
    result = {}
    for k, v in x.items():
        if isinstance(v, dict):
            result.update(flatten_dict(v, prefix=prefix + k + '_'))
        else:
            result[prefix + k] = v
    return result
#%%
if __name__ == '__main__':
    run_id = datetime.datetime.now().isoformat()

    survey_seq = get_survey_seq(batch_code)
    survey_names = {name for name, idx in survey_seq}
    surveys = {name: pd.read_csv(
        os.path.join(root_path, 'surveys', name+'_responses.csv'),
        header=1, parse_dates=['StartDate', 'EndDate'])
        for name in survey_names}

    all_log_analyses = {}
#    participants = args.participants
    assert len(participants) == len(set(participants))

    all_survey_data = []
    participant_level_data_raw = []
    excluded = []
    non_excluded_participants = []
    for participant in participants:
        log_analyses = run_log_analysis(participant)
        all_log_analyses[participant] = log_analyses
        conditions = log_analyses['conditions']
        survey_data = {name: survey[survey['clientId'] == participant].to_dict(orient='records')#.to_json(orient='records'))
            for name, survey in surveys.items()}


        print('\n'*10)
        print(participant)
        print('='*80)
        skipped_cols = set()
        complete = True
        for survey, idx in get_survey_seq(batch_code):
            print(f"\n\n{survey} {idx}")
            try:
                data = survey_data[survey][idx]
            except IndexError:
                print("MISSING!?!?!")
                complete = False
                break
            base_data = {}
            if survey in ['postFreewrite', 'postTask', 'postTask3']:
                base_data['condition'] = conditions[idx]
            elif survey in ['postExp', 'postExp3']:
                base_data['condition'] = ','.join(conditions)
            else:
                assert survey == 'intro', survey
            print('-'*20)
            for k, v in extract_survey_data(survey, data):
                print(k, ':', v)
                print()
                all_survey_data.append(dict(base_data,
                        participant_id=participant, survey=survey, idx=idx, name=k, value=v))

        if not complete:
            # Skip incomplete experiments.
            excluded.append((participant, 'incomplete'))
            continue

        non_excluded_participants.append(participant)

        base_datum = dict(participant_id=participant,
                     conditions=','.join(conditions),
                     config=log_analyses['config'])
        base_datum['dur_75'] = np.percentile(log_analyses['sug_gen_durs'], 75)
        base_datum['dur_95'] = np.percentile(log_analyses['sug_gen_durs'], 95)
        if base_datum['dur_75'] > .75:
            excluded.append(participant)
            continue
        for page, page_data in log_analyses['byExpPage'].items():
            datum = base_datum.copy()
            if '-' in page:
                kind, num = page.split('-')
            else:
                kind = page
                num = '0'
            num = int(num)
            datum['kind'] = kind
            datum['block'] = num
            datum.update(flatten_dict(log_analyses['blocks'][num]))
            classified_events = [classify_annotated_event(evt) for evt in page_data['annotated']]
            transitions = Counter(zip(itertools.chain(['start'], classified_events), itertools.chain(classified_events, ['end']))).most_common()
            for (a, b), count in transitions:
                datum[f'x_{a}_{b}'] = count
            for typ, count in Counter(classified_events).items():
                if typ is not None:
                    datum[f'num_{typ}'] = count
            participant_level_data_raw.append(datum)


    all_survey_data = pd.DataFrame(all_survey_data)

#    all_survey_data.to_csv(f'data/surveys/surveys_{batch_code}_{run_id}.csv', index=False)
    participant_level_data = pd.DataFrame(participant_level_data_raw).fillna(0)
    if 'prewriteText' in participant_level_data.columns:
        participant_level_data['prewriteLen'] = participant_level_data.prewriteText.str.len()
    participant_level_data['finalLen'] = participant_level_data.finalText.str.len()
    participant_level_data.to_csv(f'data/by_participant/participant_level_{batch_code}_{run_id}.csv', index=False)
    participant_level_data.query('kind == "final"').to_json(f'data/by_participant/participant_level_{batch_code}_{run_id}.json', orient='records')
#    print('excluded:', excluded)

#    with open(f'data/analysis_{batch_code}_{run_id}.pkl','wb') as f:
#        pickle.dump([{k: all_log_analyses[k] for k in non_excluded_participants}, survey_data], f, -1)
#        print("Wrote", f'data/analysis_{batch_code}_{run_id}.pkl')


#    all_survey_data[all_survey_data.value.str.len() > 5].to_csv(f'data/survey_freetexts_{batch_code}_{run_id}.csv')

#%%
def summarize_freetexts(all_survey_data):
    text_lens = all_survey_data.groupby(['survey', 'name']).value.apply(lambda x: x.str.len().max())
    answers_to_summarize = text_lens[text_lens>20]
    for pid, data_to_summarize in pd.merge(all_survey_data, answers_to_summarize.to_frame('mrl').reset_index(), left_on=['survey', 'name'], right_on=['survey', 'name']).groupby('participant_id'):
        print(pid)
        for (survey, idx), this_survey_data in data_to_summarize.groupby(['survey', 'idx']):
            if survey == 'intro':
                continue
            print(survey, idx)
            for row in this_survey_data.itertuples():
                print(row.condition, row.name, '===', row.value)
            print()
        print()
summarize_freetexts(all_survey_data)
#%%
def tokenize(text):
    import nltk
    return '\n'.join(' '.join(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(text))

def analyze_texts():
    participant_level_data['tokenized'] = participant_level_data.finalText.apply(tokenize)
    #%%
    from suggestion import analyzers
    wordfreq_analyzer = analyzers.WordFreqAnalyzer.build()
    #%%

    #%%
    pld = pd.concat([
            participant_level_data,
            participant_level_data.tokenized.apply(lambda doc: pd.Series(wordfreq_analyzer(doc))),
            participant_level_data.finalText.apply(lambda doc: wp_analyzer(doc)).to_frame('dist_from_best'),
            ], axis=1)
    pld[pld.kind != 'practice'].to_csv(f'data/by_participant/participant_level_{batch_code}_{run_id}_analyzed.csv', index=False)

#%%

#topic_self_reports = all_survey_data[all_survey_data.name.str.startswith('How much did you say about each topic')]
decoded_self_reports = pd.concat([
    all_survey_data,
    all_survey_data.name.str.extract(r'How much did you say about each topic.*\.\.\.-(?P<topic_name>.+)-Review (?P<review_idx>\d).*', expand=True),
    all_survey_data.name.str.extract(r'Roughly how many lines of each review were\.\.\.-(?P<sentiment>.+)\?-Review (?P<review_idx>\d).*', expand=True),
    ], axis=1)
self_report_sentiment = (
        decoded_self_reports[~decoded_self_reports.sentiment.isnull()].loc[:,['participant_id', 'condition','sentiment','review_idx', 'value']]
        .dropna(axis=1, how='all')
        .fillna({'value': 0}))
#self_report_sentiment['value'] = pd.to_numeric(self_report_sentiment['value'])
self_report_sentiment = pd.to_numeric(self_report_sentiment.set_index(['participant_id', 'condition', 'review_idx', 'sentiment']).value).unstack().reset_index()
#srs_conditions = self_report_sentiment.condition.str.split(',', expand=True)
#srs_conditions.columns = pd.Index([f'c{i}' for i in range(1,4)])

self_report_sentiment = pd.concat([
    self_report_sentiment,
#    srs_conditions
    self_report_sentiment.apply(lambda row: row['condition'].split(',')[int(row['review_idx'])-1], axis=1).to_frame('rated_condition'),
    ], axis=1)
self_report_sentiment['total_sent'] = self_report_sentiment['positive'] + self_report_sentiment['negative']
self_report_sentiment['positive'] = self_report_sentiment['positive'] / self_report_sentiment['total_sent']
self_report_sentiment['negative'] = self_report_sentiment['negative'] / self_report_sentiment['total_sent']
self_report_sentiment['imbalance'] = np.abs(self_report_sentiment['positive'] - self_report_sentiment['negative'])
self_report_sentiment.to_csv(f'data/self_report_sentiment_{batch_code}.csv', index=False)
#%%
self_report_topic = (
        decoded_self_reports[decoded_self_reports.topic_name.notnull()].loc[:, ['participant_id', 'condition', 'review_idx', 'topic_name', 'value']]
        .dropna(axis=1, how='all')
        .fillna({'value': 0}))
self_report_topic = pd.concat([
        self_report_topic,
        self_report_topic.apply(lambda row: row['condition'].split(',')[int(row['review_idx'])-1], axis=1).to_frame('rated_condition'),
    ], axis=1)
self_report_topic['value'] = pd.to_numeric(self_report_topic['value'])
topic_data = self_report_topic.set_index(['participant_id', 'condition', 'review_idx', 'topic_name']).value.unstack()
topic_data = pd.concat([
        topic_data,
        topic_data.apply(lambda row: row.name[1].split(',')[int(row.name[2])-1], axis=1).to_frame('rated_condition'),
        pd.DataFrame(dict(
                breadth=self_report_topic.groupby(['participant_id', 'condition', 'review_idx']).value.apply(lambda x: np.sum(x > 1)),
                mentions=self_report_topic.groupby(['participant_id', 'condition', 'review_idx']).value.apply(lambda x: np.sum(x > 0)),
                depth=self_report_topic.groupby(['participant_id', 'condition', 'review_idx']).value.apply(lambda x: x[x>0].mean())))
        ], axis=1)
topic_data.to_csv(f'data/self_report_topic_{batch_code}.csv')
#%%
def extract_weighted_survey_results(all_survey_data, survey, items):
    return (all_survey_data[all_survey_data.survey == survey]
            .set_index(['participant_id', 'condition', 'idx', 'name'])
            .value.unstack(level=-1)
            .apply(lambda x: sum(weight*x[col] for weight, col in items), axis=1))
#%%
if False:
    #%%
    all_survey_data = pd.DataFrame(all_survey_data)
#%%
    opinions_suggs_postTask = [(1, 'suggs-were interesting'),
        (1, 'suggs-were relevant'),
        (-1, 'suggs-were distracting'),
        (1, 'suggs-were helpful'),
        (1, "suggs-made me say something I didn't want to say."),
        (-1, 'suggs-felt pushy.'),
        (1, 'suggs-were inspiring.'),
        (1, 'suggs-were timely.')]

    opinions_suggs_postFreewrite = [(1, 'suggs-gave me ideas about what to write about.'),
        (-1, 'suggs-made it harder to come up with my own ideas about what to write about.'),
        (1, 'suggs-gave me ideas about words or phrases to use.'),
        (-1, 'suggs-made it harder for me to come up with my own words or phrases.'),
        (1, 'suggs-were interesting.'),
        (-1, 'suggs-were distracting.'),
        (1, 'suggs-were relevant.'),
        (-1, "suggs-didn't make sense."),
        (-1, 'suggs-were repetitive.'),
        (1, 'suggs-were inspiring.')]

    pd.DataFrame(dict(
        opinion_of_suggs_posttask=extract_weighted_survey_results(all_survey_data, 'postTask', opinions_suggs_postTask),
        opinion_of_suggs_postfreewrite=extract_weighted_survey_results(all_survey_data, 'postFreewrite', opinions_suggs_postFreewrite)),
    ).to_csv('data/opinion_of_suggs.csv')
#%%

    extract_weighted_survey_results(all_survey_data, 'postTask', [(1, 'final-I knew what I wanted to say.'),
        (1, 'final-I knew how to say what I wanted to say.'),
        (-1, 'final-I often paused to think of what to say.'),
        (-1, 'final-I often paused to think of how to say what I wanted to say.'),
        (1, 'final-I was able to express myself fluently.'),
        (-1, 'final-I had trouble expressing myself fluently.')]).to_frame('expression').to_csv('data/expression.csv')


def analyze_sentiment_change():
    post_task = all_survey_data[all_survey_data.survey == 'postTask']
    results = []
    for participant in non_excluded_participants:
        stars_post = post_task.set_index(['participant_id', 'condition', 'idx', 'name']).value.unstack(level=-1).loc[participant, "Now that you've had a chance to write about it, how many stars would you give your experience at...-&nbsp;"].to_frame('stars_post')
        log = all_log_analyses[participant]
        places = pd.DataFrame([block['place'] for block in log['blocks']])
        places.index.name = 'idx'
    #    places['condition'] = log['conditions']
        results.append(
                pd.merge(stars_post.reset_index(level=0), places.rename(columns={'stars': 'stars_pre'}), left_index=True, right_index=True, how='outer')
                .assign(participant_id=participant).reset_index())
    pd.concat(results, ignore_index=True).to_csv('data/sentiment_change.csv')

#%%
def analyze_personality():
    personality_weights = dict(
    extraversion=[(1, 'pers-I am the life of the party.'),
        (1, 'pers-I talk to a lot of different people at parties.'),
        (-1, "pers-I don't talk a lot."),
        (-1, 'pers-I keep in the background.')],
    agreeableness=[
    (1, "pers-I sympathize with others' feelings."),
    (1, "pers-I feel others' emotions."),
    (-1, 'pers-I am not really interested in others.'),
    (-1, "pers-I am not interested in other people's problems."),],

    conscientiousness=[
    (1, 'pers-I get chores done right away.'),
    (1, 'pers-I like order.'),
    (-1, 'pers-I often forget to put things back in their proper place.'),
    (-1, 'pers-I make a mess of things.'),],

    neuroticism=[
    (1, 'pers-I have frequent mood swings.'),
    (1, 'pers-I get upset easily.'),
    (-1, 'pers-I am relaxed most of the time.'),
    (-1, 'pers-I seldom feel blue.'),],

    imagination=[
    (1, 'pers-I have a vivid imagination.'),
    (-1, 'pers-I have difficulty understanding abstract ideas.'),
    (-1, 'pers-I am not interested in abstract ideas.'),
    (-1, 'pers-I do not have a good imagination. ')],

    NFC=[
    (1, 'pers-I would prefer complex to simple problems.'),
    (1, 'pers-I like to have the responsibility of handling a situation that requires a lot of thinking.'),
    (-1, 'pers-Thinking is not my idea of fun.'),
    (-1, 'pers-I would rather do something that requires little thought than something that is sure to challenge my thinking abilities.'),])

    pd.DataFrame({k: extract_weighted_survey_results(all_survey_data, 'postExp', v) for k, v in personality_weights.items()}).to_csv('personality.csv')
