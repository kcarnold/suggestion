import pickle
import gzip
from sys import intern
import ujson as json
import tqdm
import joblib
from suggestion.tokenization import tokenize
from suggestion.suffix_array import DocSuffixArray

reviews = []
businesses = {}
with gzip.open('/Users/kcarnold/Data/Yelp/yelp_academic_dataset.json.gz', 'rb') as f:
    for line in f:
        rec = json.loads(line.decode('utf8'))
        if rec['type'] == 'review':
            reviews.append(rec)
        elif rec['type'] == 'business':
            businesses[rec['business_id']] = rec

all_restaurants = {
    b['business_id']: b for b in businesses.values()
    if 'Restaurants' in b['categories'] and b['open']}
restaurant_reviews = [r for r in reviews if r['business_id'] in all_restaurants]

print("Dumping un-tokenized as JSON")
with open('models/reviews.json', 'w') as f:
    json.dump(restaurant_reviews, f)

bad_eoses = ["coffee roasting co . </S> <S> is one of the best coffee shops", "gogi ! </S> <S>", "green st . </S> <S> is one of my standard go to"]

import re
cant_type = re.compile(r'[^\-a-z., !\']')


def tokenized_review(text):
    text = text.lower()
    text = cant_type.sub(' ', text)
    line = ' '.join(tokenize(text)[0])
    line = line.replace('<D> <P> <S>', '<D>')
    line = line.replace('<P> <S>', '<P>')
    for bad_eos in bad_eoses:
        if bad_eos in line:
            print("Subbing", bad_eos)
            line = line.replace(bad_eos, bad_eos.replace('</S> <S> ', ''))
    return [intern(word) for word in line.split()]

tokenized_reviews = [tokenized_review(review['text']) for review in tqdm.tqdm(restaurant_reviews, desc="Tokenizing")]
print("Saving reviews")
with open('models/tokenized_reviews.pkl', 'wb') as f:
    pickle.dump(tokenized_reviews, f, -1)

with open('models/yelp_train.txt', 'w') as fp_train, open('models/yelp_test.txt', 'w') as fp_test, open('models/yelp-char.txt', 'w') as f_char:
    for i, review in enumerate(tqdm.tqdm(tokenized_reviews, desc="Writing")):
        line = ' '.join(review)
        if i % 10 == 0:
            print(line, file=fp_test)
        else:
            print(line, file=fp_train)
#        print(' '.join(text.replace(' ', '_')), file=f_char)


print("Build suffix array")
sufarr = DocSuffixArray.construct(tokenized_reviews)

joblib.dump(dict(suffix_array=sufarr.suffix_array, doc_idx=sufarr.doc_idx, tok_idx=sufarr.tok_idx, lcp=sufarr.lcp), 'models/yelp_sufarr.joblib')
