import os
try:
    import ujson as json
except ImportError:
    import json
import re
import numpy as np
from suggestion.util import mem
from suggestion.paths import paths
import subprocess

#
# Data for decoding surveys.
#

skip_col_re = re.compile(
    r'Great.job|Q_\w+|nextURL|clientId|Timing.*|Browser.*|Location.*|Recipient.*|Response.+|ExternalDataReference|Finished|Status|IPAddress|StartDate|EndDate|Welcome.+|Display Order|Demographic Questions|Closing Survey.+|revisionDesc|prewrite')

prefix_subs = {
    "How much do you agree with the following statements about the suggestions that the system gave?-They ": "suggs-",
    "How much do you agree with the following statements?-The suggestions ": "suggs-",
    "How much do you agree with the following statements about the words or phrases that the keyboard...-They ": "suggs-",
    "Now think about the brainstorming you did before the final writing. How much do you agree with th...-": "brainstorm-",
    "Think about when you were typing out your ${e://Field/revisionDesc}. How much do you agree with t...-": "final-",
    "How Accurately Can You Describe Yourself? Describe yourself as you generally are now, not as you...-": "pers-",
    "Describe yourself as you generally are now, not as you wish to be in the future. Describe yoursel...-": "pers-",
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


def get_existing_requests_raw(logfile):
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
            phrases.append('')
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
            phrases=phrases,
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

@mem.cache()
def get_existing_reqs_cached_raw(*a, **kw):
    return json.dumps(get_existing_requests_raw(*a, **kw))

def get_existing_requests(*a, **kw):
    return json.loads(get_existing_reqs_cached_raw(*a, **kw))



def get_rev(participant):
    logpath = paths.parent / 'logs' / (participant+'.jsonl')
    with open(logpath) as logfile:
        for line in logfile:
            line = json.loads(line)
            if 'rev' in line:
                return line['rev']

def checkout_old_code(git_rev):
    import shutil
    by_rev = paths.parent / 'old-code'
    rev_root = by_rev / git_rev
    if not os.path.isdir(rev_root):
        print("Checking out repository at", git_rev)
        subprocess.check_call(['git', 'clone', '..', git_rev], cwd=by_rev)
        subprocess.check_call(['git', 'checkout', git_rev], cwd=rev_root)
        print("Installing npm packages")
        subprocess.check_call(['yarn'], cwd=os.path.join(rev_root, 'frontend'))


@mem.cache
def get_log_analysis_raw(participant, git_rev=None):
    logpath = paths.parent / 'logs' / (participant+'.jsonl')
    if git_rev is None:
        git_rev = get_rev(participant)
    checkout_old_code(git_rev)
    analyzer_path = os.path.join(paths.parent, 'frontend', 'analysis')
    with open(logpath) as logfile:
        return subprocess.check_output([analyzer_path], stdin=logfile), git_rev


def get_log_analysis(participant, git_rev=None):
    result, git_rev = get_log_analysis_raw(participant, git_rev=git_rev)
    analyzed = json.loads(result)
    analyzed['git_rev'] = git_rev
    return analyzed



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



def group_requests_by_session(suggestion_data_raw, participant_id):
    '''Given a set of raw suggestions, group them by experiment page.'''
    prev_request_id = None
    res = []
    cur_block = []
    for sugg in suggestion_data_raw:
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
        cur_block.append(dict(sugg))
    res.append(cur_block)
    return res

def get_content_stats_single_suggestion(sugg, word_freq_analyzer):
    from suggestion import suggestion_generator
    meta = {k: sugg.pop(k, None) for k in 'domain sentiment_method sentiment temperature useSufarr use_bos_suggs'.split()}

    if not meta['domain'].startswith('yelp'):
        return

    if sugg['cur_word']:
        # Skip partial words.
        return

    model = suggestion_generator.get_or_load_model(meta['domain'])
    try:
        toks = suggestion_generator.tokenize_sofar(sugg['sofar'])
    except:
        # Tokenization failed.
        return
    # Optimization: trim context to the n-gram level, plus some padding.
    toks = toks[-10:]
    state = model.get_state(toks)[0]
    clf_startstate = suggestion_generator.sentiment_classifier.get_state(toks)
    res = []
    for sugg_slot, phrase in enumerate(sugg['phrases']):
        if phrase:
            sentiment_posteriors = suggestion_generator.sentiment_classifier.classify_seq_by_tok(clf_startstate, phrase)
            sentiment = np.mean(sentiment_posteriors, axis=0) @ suggestion_generator.sentiment_classifier.sentiment_weights
        else:
            sentiment = None
        analyzer_indices = [word_freq_analyzer.word2idx.get(tok) for tok in phrase]
        res.append(dict(
            request_id=sugg['request_id'],
            sugg_slot=sugg_slot,
            sugg_contextual_llk=model.score_seq(state, phrase)[0],
            sugg_unigram_llk=np.nanmean(np.array([word_freq_analyzer.log_freqs[idx] if idx is not None else np.nan for idx in analyzer_indices])),
            sugg_sentiment=sentiment))
    return res

