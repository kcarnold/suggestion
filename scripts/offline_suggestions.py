import json
import argparse
import time
import traceback
from suggestion import suggestion_generator
import pandas as pd
import numpy as np

import re
import os
import argparse
import pathlib
import pickle
import subprocess
import pandas as pd
import json
import numpy as np
import datetime
import itertools
import yaml

batch_code = 'sent3_01'

if '__file__' in globals():
    root_path = pathlib.Path(__file__).resolve().parent.parent
else:
    root_path = pathlib.Path('~/code/suggestion').expanduser()

participants = yaml.load(open(root_path / 'participants.yaml'))[batch_code].split()
assert len(participants) == len(set(participants)), "Duplicate participants"

def do_request_raw(request):
    return suggestion_generator.get_suggestions(
        sofar=request['sofar'], cur_word=request['cur_word'],
        **suggestion_generator.request_to_kwargs(request))

def do_request(request):
    start = time.time()
    try:
        phrases = do_request_raw(request)
    except Exception:
        traceback.print_exc()
        phrases = None
    dur = time.time() - start
    return dur, phrases

#%%
def get_existing_requests(logfile):
    requests = []
    responses = []
    for line in open(logfile):
        if not line:
            continue
        entry = json.loads(line)
        if entry['kind'] == 'meta' and entry['type'] == 'requestSuggestions':
            requests.append(entry['request'].copy())
        if entry['type'] == 'receivedSuggestions':
            responses.append(dict(entry['msg'].copy(), responseTimestamp=entry['jsTimestamp']))
    assert len(requests) == len(responses)
    suggestions = []
    for request, response in zip(requests, responses):
        phrases = response['next_word']
        phrases = [' '.join(phrase['one_word']['words'] + phrase['continuation'][0]['words']) for phrase in phrases]
        while len(phrases) < 3:
            phrases.append([''])
        assert len(phrases) == 3
        p1, p2, p3 = phrases
        request_ts = request.pop('timestamp')
        assert request_ts == response.pop('timestamp') # the server sends the client request timestamp back to the client...
        response_ts = response.pop('responseTimestamp')
        latency = response_ts - request_ts
        ctx = request['sofar'] + ''.join(ent['letter'] for ent in request['cur_word'])
        sentiment = request.get('sentiment', 'none')
        if sentiment in [1,2,3,4,5]:
            sentiment = 'match'
        entry = dict(
            **request,
            ctx=ctx,
            p1=p1, p2=p2, p3=p3,
            server_dur=response['dur'],
            latency=latency,
            sentiment_method=sentiment,
            timestamp=request_ts)
        suggestions.append(entry)
    return suggestions
suggestion_data_raw = {participant: get_existing_requests(root_path / 'logs' / f'{participant}.jsonl') for participant in participants}



#%%
def redo_requests(requests, flags={}):
    results = []
    durs = []
    for request in requests:
        dur, phrases = do_request(request)
        results.append(phrases)
        durs.append(dur)
#        print('{prefix}\t{curWord}\t{dur:.2f}\t{phrases}'.format(
#            prefix=request['sofar'], curWord=request['cur_word'],
#            dur=dur, phrases=json.dumps(phrases)))
    return results, durs

#%%
%prun redo_requests([req for req in suggestion_data_raw['061563'] if req['domain'] != 'sotu'][:10])
#%%

req = [req for req in suggestion_data_raw['061563'] if req['domain'] != 'sotu'][0]
%timeit redo_requests([req])
#%%
{participant_id: len(requests) for participant_id, requests in suggestion_data_raw.items()}

#%% Analyze sentiment of suggestions made.
import tqdm
clf = suggestion_generator.sentiment_classifier
sentiment_results = []
for participant_id, requests in suggestion_data_raw.items():
    for request_seq, req in enumerate(tqdm.tqdm(requests, desc=participant_id)):
        if req['domain'] == 'sotu':
            continue
        if req['continuation_length'] < 5 and len(req['cur_word']) == 0:
            # This suggestion was never even shown.
            continue
        try:
            toks = suggestion_generator.tokenize_sofar(req['sofar'])
        except:
            continue
        phrases = [req[f'p{i}'] for i in range(1,4)]
        phrases = [p for p in phrases if p]
        if not phrases: continue
        clf_state = clf.get_state(toks[-6:], bos=False)
        posteriors = [np.mean(clf.classify_seq_by_tok(clf_state, phrase.split()), axis=0) for phrase in phrases]
#        for phrase_idx, phrase in enumerate(phrases):
#            posterior =
#            sentiment = posterior @ clf.sentiment_weights
#            datum = dict(
#                    participant_id=participant_id,
#                    request_seq=request_seq,
#                    sentiment_method=req['sentiment_method'],
#                    request_sentiment=req.get('sentiment', 'none'),
#                    phrase_idx=phrase_idx,
#                    sentiment=sentiment,
#                    posterior=posterior)
#            for i in range(5):
#                datum[f'post{i+1}'] = posterior[i]
#            sentiment_results.append(datum)
        sentiment_results.append(dict(
            participant_id=participant_id,
            request_seq=request_seq,
            sentiment_method=req['sentiment_method'],
            request_sentiment=req.get('sentiment', 'none'),
            posteriors=posteriors,
            sentiments=posteriors @ clf.sentiment_weights,
            cur_word_len=len(req['cur_word'])))
#%%
sentiment_df = pd.DataFrame(sentiment_results)
sentiment_df['full_word'] = sentiment_df.cur_word_len == 0
#%% Simplest metric: sentiment variance across suggestions.
sentiment_df['sentiment_std'] = sentiment_df.sentiments.map(lambda x: np.std(x))
sentiment_df.groupby(['request_sentiment', 'full_word']).sentiment_std.describe()
#%% Or -- range of sentiments.
sentiment_df['sentiment_ptp'] = sentiment_df.sentiments.map(lambda x: np.ptp(x))
sentiment_df.groupby(['request_sentiment', 'full_word']).sentiment_ptp.describe()
#%%
from suggestion.diversity import scalar_mean_pdist_diversity, scalar_dpp_diversity
sentiment_df['sentiment_mpd'] = sentiment_df.sentiments.map(scalar_mean_pdist_diversity)
sentiment_df['sentiment_dpp'] = sentiment_df.sentiments.map(scalar_dpp_diversity)
#%%
#sentiment_df.query('full_word').groupby(['request_sentiment']).sentiment_dpp.describe()
sentiment_df.groupby(['full_word', 'request_sentiment']).sentiment_std.describe()
#%% Cross-check.
do_request(requests[-10])
#%%
requests[-10]
#%%
req_meta = []
for participant_id, requests in suggestion_data_raw.items():
    for request_seq, req in enumerate(tqdm.tqdm(requests, desc=participant_id)):
        if req['domain'] == 'sotu':
            continue
        if req['continuation_length'] < 5 and len(req['cur_word']) == 0:
            # This suggestion was never even shown.
            continue
        try:
            toks = suggestion_generator.tokenize_sofar(req['sofar'])
        except:
            continue
        phrases = [req[f'p{i}'] for i in range(1,4)]
        phrases = [p for p in phrases if p]
        if not phrases: continue
        req_meta.append(dict(
            req,
            toks=toks,
            participant_id=participant_id,
            request_seq=request_seq,
            sofar=req['sofar'],
            phrases=phrases,))
#%%
pd.concat([sentiment_df, pd.DataFrame(req_meta)], axis=1).to_csv('all_sentiments-clf-order2.csv', index=False)

#%%
rs = np.random.RandomState(0)
samples = []
while len(samples) < 20:
    idx = rs.choice(len(sentiment_results))
    if idx in samples or sentiment_results[idx]['cur_word_len'] > 0:
        continue
    samples.append(idx)
#%%
to_rate = []
for idx in samples:
    meta = req_meta[idx]
    resp = sentiment_results[idx]
    print(            meta['sofar'][-50:].replace('\n', ' '))
    to_rate.append((
            idx,
            meta['sofar'][-50:].replace('\n', ' '),
            meta['p1'],
            meta['p2'],
            meta['p3'],))
if False:
    pd.DataFrame(to_rate, columns=['idx', 'context', 's1', 's2', 's3']).to_csv('to_rate_sentiment_suggs.csv', index=False)
#%%
rating_results = pd.read_csv('to_rate_sentiment_suggs.csv')
#%%
pivoted_ratings = pd.concat({i: rating_results.loc[:,['idx', f's{i}', f'pos{i}', f'neg{i}', f'sensible{i}']].rename(columns={f's{i}': 'sug', f'pos{i}': 'pos', f'neg{i}': 'neg', f'sensible{i}': 'sensible'}) for i in range(1,4)}, axis=0, names=['sug_idx', 'req_idx'])
pivoted_ratings = pivoted_ratings.reset_index(level=0)
#%%
pivoted_ratings.pos + pivoted_ratings.neg
#%%
def to_sentiment_dist(pos, neg):
    dist = np.ones(5)
    dist[3:] += pos
    dist[:2] += neg
    dist /= dist.sum()
    return dist
to_sentiment_dist(2, 1)
#%%
model = suggestion_generator.get_model('yelp_train-balanced')
#%%
from scipy.special import kl_div
rating_match = []
for row in pivoted_ratings.itertuples():
    phrase_idx = row.sug_idx - 1
    posterior = sentiment_results[row.idx]['posteriors'][phrase_idx]
    sent_dist = to_sentiment_dist(row.pos/2, row.neg/2)
    sent_dist_null = to_sentiment_dist(0, 0)
    ks_dist_null = np.max(np.abs(posterior - to_sentiment_dist(row.neg, row.pos)))
    ks_dist = np.max(np.abs(posterior - sent_dist))
    rating_sentiment = .5 + (row.pos - row.neg) / 4

    meta = req_meta[row.idx]
    sofar = meta['sofar'][-50:].replace('\n', ' ')
    phrase = meta['phrases'][phrase_idx]

    # Sensibility
    ctx = model.get_state(meta['toks'], bos=False)[0]
    score = np.mean(model.score_seq_by_word(ctx, phrase.split()))
    null_score = np.mean(model.score_seq_by_word(model.null_context_state, phrase.split()))
    unigram_score = np.mean(model.unigram_probs[[model.model.vocab_index(tok) for tok in phrase.split()]])

    rating_match.append(dict(
            idx=row.idx, sug_idx=row.sug_idx,
            ks_dist_null=ks_dist_null, ks_dist=ks_dist,
            kl_div=np.sum(kl_div(posterior, sent_dist)),
            kl_div_inv=np.sum(kl_div(sent_dist, posterior)),
            kl_div_null=np.sum(kl_div(posterior, sent_dist_null)),
            clf_sentiment=posterior @ clf.sentiment_weights,
            rating_sentiment=rating_sentiment,
            score=score,
            null_score=null_score,
            pmi=score - null_score,
            unigram_score=unigram_score,
            sensible=row.sensible,
            sofar=sofar,
            phrase=phrase,
            ))
rating_match_df = pd.DataFrame(rating_match)
rating_match_df.to_csv('rating_match-clf-order2.csv',index=False)
#%%
2**rating_match_df.kl_div.mean(), 2**rating_match_df.kl_div_null.mean()
#%%
(rating_match_df.ks_dist_null - rating_match_df.ks_dist).mean()
#%%
from scipy.stats import pearsonr
pearsonr(rating_match_df.sensible, rating_match_df.pmi)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

x = rating_match_df.sensible
y = rating_match_df.pmi

if False:
    # Use JointGrid directly to draw a custom plot
    grid = sns.JointGrid(x, y, space=0, size=6, ratio=50)
    grid.plot_joint(plt.scatter, color="g")
    grid.plot_marginals(sns.rugplot, height=1, color="g")
else:
    sns.factorplot(data=rating_match_df, x='sensible', y='pmi')
#%%
#plt.scatter(rating_match_df.rating_sentiment, rating_match_df.clf_sentiment)
sns.regplot(data=rating_match_df, x='rating_sentiment', y='clf_sentiment')
