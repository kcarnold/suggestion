import gzip
import json

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

from megacomplete.tokenize import tokenize
bad_eoses = ["coffee roasting co . </S> <S> is one of the best coffee shops", "gogi ! </S> <S>", "green st . </S> <S> is one of my standard go to"]

import re
cant_type = re.compile(r'[^\-a-z., !\']')

with open('models/yelp_train.txt', 'w') as fp_train, open('models/yelp_test.txt', 'w') as fp_test, open('models/yelp-char.txt', 'w') as f_char:
    for i, review in enumerate(restaurant_reviews):
        text = review['text'].lower()
        text = cant_type.sub(' ', text)
        line = ' '.join(tokenize(text)[0])
        line = line.replace('<D> <P> <S>', '<D>')
        line = line.replace('<P> <S>', '<P>')
        for bad_eos in bad_eoses:
            if bad_eos in line:
                print("Subbing", bad_eos)
                line = line.replace(bad_eos, bad_eos.replace('</S> <S> ', ''))
        if i % 10 == 0:
            print(line, file=fp_test)
        else:
            print(line, file=fp_train)
        print(' '.join(text.replace(' ', '_')), file=f_char)