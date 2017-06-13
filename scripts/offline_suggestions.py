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
    for request in requests:
        dur, phrases = do_request(request)
        results.append((request, dur, phrases))
#        print('{prefix}\t{curWord}\t{dur:.2f}\t{phrases}'.format(
#            prefix=request['sofar'], curWord=request['cur_word'],
#            dur=dur, phrases=json.dumps(phrases)))

#%%
%prun redo_requests([req for req in suggestion_data_raw['061563'] if req['domain'] != 'sotu'][:10])
#%%

req = [req for req in suggestion_data_raw['061563'] if req['domain'] != 'sotu'][0]
%timeit redo_requests([req])