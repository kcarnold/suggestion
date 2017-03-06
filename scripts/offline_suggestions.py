import json
import argparse
import time
from suggestion import suggestion_generator

def do_request(request):
    start = time.time()
    # copy-and-paste from app.py, somewhat yuk but whatever.
    phrases = suggestion_generator.get_suggestions(
                    request['sofar'], request['cur_word'],
                    domain=request.get('domain', 'yelp_train'),
                    rare_word_bonus=request.get('rare_word_bonus', 1.0),
                    use_sufarr=request.get('useSufarr', False),
                    temperature=request.get('temperature', 0.))
    dur = time.time() - start
    return dur, phrases

parser = argparse.ArgumentParser()
parser.add_argument('source')
args = parser.parse_args()

entries = (json.loads(line) for line in open(args.source) if line)
requests = [entry['request'] for entry in entries if entry['kind'] == 'meta' and entry['type'] == 'requestSuggestions']

results = []
for request in requests:
    dur, phrases = do_request(request)
    results.append((request, dur, phrases))
    print('{prefix}\t{curWord}\t{dur:.2f}\t{phrases}'.format(
        prefix=request['sofar'], curWord=request['cur_word'],
        dur=dur, phrases=json.dumps(phrases)))
