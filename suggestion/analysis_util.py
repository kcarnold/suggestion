import json

def get_existing_requests(logfile):
    # Each request has a timestamp, which is effectively its unique id.
    weird_counter = 0
    dupe_counter = 0
    requests = {}
    for line in open(logfile):
        if not line:
            continue
        entry = json.loads(line)
        if entry['kind'] == 'meta' and entry['type'] == 'requestSuggestions':
            msg = entry['request'].copy()
            requests[msg['timestamp']] = {'request': msg, 'response': None}

        if entry['type'] == 'receivedSuggestions':
            msg = dict(entry['msg'], responseTimestamp=entry['jsTimestamp'])
            if requests[msg['timestamp']]['response'] is not None:
                weird_counter += 1
            requests[msg['timestamp']]['response'] = msg

    # Keep only responded-to requests
    requests = {ts: req for ts, req in requests.items() if req['response'] is not None}

    def without_ts(dct):
        dct = dct['request'].copy()
        dct.pop('timestamp')
        return dct
    requests_dedupe = []
    for ts, request in sorted(requests.items()):
        if len(requests_dedupe) > 0 and without_ts(request) == without_ts(requests_dedupe[-1]):
            dupe_counter += 1
        else:
            requests_dedupe.append(request)

    suggestions = []
    prev_request_ts = None
    for entry in requests_dedupe:
        request = entry['request']
        response = entry['response']
        phrases = response['next_word']
        phrases = [' '.join(phrase['one_word']['words'] + phrase['continuation'][0]['words']) for phrase in phrases]
        while len(phrases) < 3:
            phrases.append([''])
        if len(phrases) > 3:
            weird_counter += 1
            phrases = phrases[:3]
        p1, p2, p3 = phrases
        request_ts = request.pop('timestamp')
        response_request_ts = response.pop('timestamp')
        assert request_ts == response_request_ts # the server sends the client request timestamp back to the client...
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
            server_dur=response['dur'] * 1000,
            latency=latency,
            time_since_prev=request_ts - prev_request_ts if prev_request_ts is not None else None,
            sentiment_method=sentiment,
            timestamp=request_ts)
        suggestions.append(entry)
        prev_request_ts = request_ts

    if weird_counter != 0 or dupe_counter != 0:
        print(f"{logfile} weird={weird_counter} dup={dupe_counter}")

    return suggestions
