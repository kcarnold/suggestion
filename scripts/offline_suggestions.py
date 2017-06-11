import json
import argparse
import time
import traceback
from suggestion import suggestion_generator
import pandas as pd
import numpy as np

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
            responses.append(entry['msg'].copy())
    assert len(requests) == len(responses)
    suggestions = []
    for request, response in zip(requests, responses):
        phrases = response['next_word']
        phrases = [' '.join(phrase['one_word']['words'] + phrase['continuation'][0]['words']) for phrase in phrases]
        assert len(phrases) == 3
        p1, p2, p3 = phrases
        request_ts = request.pop('timestamp')
        response_ts = response.pop('timestamp')
        latency = response_ts - request_ts
        ctx = request['sofar'] + ''.join(ent['letter'] for ent in request['cur_word'])
        entry = dict(
            **request,
            ctx=ctx,
            p1=p1, p2=p2, p3=p3,
            server_dur=response['dur'],
            latency=latency)
        suggestions.append(entry)
    return suggestions




def redo_requests(requests, flags={}):
    results = []
    for request in requests:
        dur, phrases = do_request(request)
        results.append((request, dur, phrases))
        print('{prefix}\t{curWord}\t{dur:.2f}\t{phrases}'.format(
            prefix=request['sofar'], curWord=request['cur_word'],
            dur=dur, phrases=json.dumps(phrases)))
