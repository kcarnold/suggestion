import json
import argparse
import time
from suggestion import suggestion_generator

parser = argparse.ArgumentParser()
parser.add_argument('source')
args = parser.parse_args()

data = json.load(open(args.source))
def do_request(request):
    start = time.time()
    prefix = request['prefix']
    curWord = request['curWord']
    phrases = suggestion_generator.get_suggestions(
        sofar=prefix, cur_word=curWord,
        domain='yelp_train', rare_word_bonus=request['rare_word_bonus'])
    dur = time.time() - start
    return dur, phrases


results = []
for request in data['requests']:
    dur, phrases = do_request(request)
    results.append((request, dur, phrases))
    print('{prefix}\t{curWord}\t{rare_word_bonus}\t{dur:.2f}\t{phrases}'.format(
        prefix=request['prefix'], curWord=request['curWord'], rare_word_bonus=request['rare_word_bonus'],
        dur=dur, phrases=json.dumps(phrases)))
