import json
import re

#
# Data for decoding surveys.
#

skip_col_re = re.compile(
    r'Great.job|Q_\w+|nextURL|clientId|Timing.*|Browser.*|Location.*|Recipient.*|Response.+|ExternalDataReference|Finished|Status|IPAddress|StartDate|EndDate|Welcome.+|Display Order|Demographic Questions|Closing Survey.+')

prefix_subs = {
    "How much do you agree with the following statements about the suggestions that the system gave?-They ": "suggs-",
    "How much do you agree with the following statements?-The suggestions ": "suggs-",
    "How much do you agree with the following statements about the words or phrases that the keyboard...-They ": "suggs-",
    "Now think about the brainstorming you did before the final writing. How much do you agree with th...-": "brainstorm-",
    "Think about when you were typing out your ${e://Field/revisionDesc}. How much do you agree with t...-": "final-",
    "How Accurately Can You Describe Yourself? Describe yourself as you generally are now, not as you...-": "pers-",
}

decode_scales = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Somewhat disagree": 3,
        "Neither agree nor disagree": 4,
        "Somewhat agree": 5,
        "Agree": 6,
        "Strongly agree": 7,

        "Very Inaccurate": 1,
        "Moderately Inaccurate": 2,
        "Neither Accurate Nor Inaccurate": 3,
        "Moderately Accurate": 4,
        "Very Accurate": 5}


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

