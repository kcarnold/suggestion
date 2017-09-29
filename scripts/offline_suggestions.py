import json
import argparse
import time
import traceback
import yaml
import tqdm
from suggestion import suggestion_generator
from suggestion.paths import paths



def do_request_raw(request):
    flags = request['flags']
    if flags.get('split'):
        return suggestion_generator.get_split_recs(request['sofar'], request['cur_word'], request['flags'])
    elif flags.get('alternatives'):
        return suggestion_generator.get_clustered_recs(request['sofar'], request['cur_word'], request['flags'])
    else:
        suggestion_kwargs = suggestion_generator.request_to_kwargs(flags)
        phrases, sug_state = suggestion_generator.get_suggestions(
            sofar=request['sofar'], cur_word=request['cur_word'],
            sug_state=None,
            **suggestion_kwargs)
    return phrases


def do_request(request):
    start = time.time()
    if True:
    # try:
        phrases = do_request_raw(request)
    # except Exception:
    #     traceback.print_exc()
    #     phrases = None
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
            responses.append(dict(entry['msg'].copy(), responseTimestamp=entry['jsTimestamp']))
    assert len(requests) == len(responses)
    return requests, responses


def redo_requests(requests):
    results = []
    durs = []
    for request in tqdm.tqdm(requests):
        dur, phrases = do_request(request)
        results.append(phrases)
        durs.append(dur)
#        print('{prefix}\t{curWord}\t{dur:.2f}\t{phrases}'.format(
#            prefix=request['sofar'], curWord=request['cur_word'],
#            dur=dur, phrases=json.dumps(phrases)))
    return results, durs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch_code')
    args = parser.parse_args()

    participants = yaml.load(open(paths.parent / 'participants.yaml'))[args.batch_code].split()
    assert len(participants) == len(set(participants)), "Duplicate participants"
    for participant in participants:
        print("Rerunning", participant)
        requests, responses = get_existing_requests(paths.parent / 'logs' / f'{participant}.jsonl')
        redo_requests(requests)
