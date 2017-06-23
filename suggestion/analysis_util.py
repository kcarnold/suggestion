import json

def get_existing_requests(logfile):
    requests = []
    responses = []
    for line in open(logfile):
        if not line:
            continue
        entry = json.loads(line)
        if entry['kind'] == 'meta' and entry['type'] == 'requestSuggestions':
            requests.append(entry['request'].copy())
            # print(f"Request {requests[-1]['request_id']}")
            if len(requests) > 1 and requests[-1]['request_id'] == requests[-2]['request_id']:
                # print("Duplicate request, keeping newer.")
                newer_timestamp = max(ent['timestamp'] for ent in requests[-2:])
                requests[-2:] = [ent for ent in requests[-2:] if ent['timestamp'] == newer_timestamp]

        if entry['type'] == 'receivedSuggestions':
            msg = entry['msg'].copy()
            responses.append(dict(msg, responseTimestamp=entry['jsTimestamp']))
            # print(f"Response {msg['request_id']}")
            if len(responses) > 1 and msg['request_id'] == responses[-2]['request_id']:
                # print("Dropping the older of duplicate responses.")
                # Keep the pair with the newer timestamp.
                newer_timestamp = max(ent['timestamp'] for ent in responses[-2:])
                responses[-2:] = [ent for ent in responses[-2:] if ent['timestamp'] == newer_timestamp]
            corresponding_request_seq = len(responses) - 1
            if msg['request_id'] != requests[corresponding_request_seq]['request_id']:
                print(f"Warning, mismatched request {msg['request_id']} vs {requests[corresponding_request_seq]['request_id']} at {len(requests)} {len(responses)}")
                assert False
                # Ok, what happened? Perhaps the JS client sent a duplicate request, and (somehow?) only one of them got a response.
                # If that's the case, the previous request was a duplicate.
                if requests[corresponding_request_seq]['request_id'] == requests[corresponding_request_seq - 1]['request_id']:
                    requests.pop(corresponding_request_seq)
    # I've occasionally seen the final request get unanswered.
    if len(requests) == len(responses) + 1:
        assert requests[-2]['request_id'] == responses[-1]['request_id']
        requests.pop(-1)
    assert len(requests) == len(responses), f"Invalid logfile? {logfile}, {len(requests)} {len(responses)}"
    suggestions = []
    prev_request_ts = None
    for request, response in zip(requests, responses):
        phrases = response['next_word']
        phrases = [' '.join(phrase['one_word']['words'] + phrase['continuation'][0]['words']) for phrase in phrases]
        while len(phrases) < 3:
            phrases.append([''])
        assert len(phrases) == 3
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
    return suggestions
