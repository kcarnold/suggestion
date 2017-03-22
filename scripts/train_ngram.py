import argparse
import pickle
from sys import intern
import tqdm
import joblib
from suggestion.util import spacy_tok_to_doc, dump_kenlm
from suggestion.suffix_array import DocSuffixArray

import re
cant_type = re.compile(r'[^\-a-z., !\']')

spelled_out = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen'.split()

def sub_numbers(txt):
    def number_form(match):
        num = int(match.group(0))
        if num < len(spelled_out):
            return spelled_out[num]
        return ''
    return re.sub(r'\b\d+\b', number_form, txt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',
                        help='Input data filename (pickle)',
                        default='yelp_preproc/train_data.pkl')
    parser.add_argument('--model-name', default="yelp_train",
                        help="model name")
    args = parser.parse_args()

    print("Loading...", flush=True)
    data = pickle.load(open(args.input, 'rb'))
    reviews = data['data']

    def convert_tokenization(doc):
        doc = doc.lower()
        doc = sub_numbers(doc)
        sents = doc.split('\n')
        sents = [cant_type.sub('', sent) for sent in sents]
        return [intern(word) for word in spacy_tok_to_doc(sents)]

    tokenized_reviews = [convert_tokenization(tokenized)
        for tokenized in tqdm.tqdm(reviews.tokenized, desc="Converting format")]

    print("Saving reviews")
    with open('models/tokenized_reviews.pkl', 'wb') as f:
        pickle.dump(tokenized_reviews, f, -1)

    dump_kenlm(args.model_name, (' '.join(doc) for doc in tqdm.tqdm(tokenized_reviews, desc="Writing")))

    sufarr = DocSuffixArray.construct(tokenized_reviews)
    joblib.dump(dict(suffix_array=sufarr.suffix_array, doc_idx=sufarr.doc_idx, tok_idx=sufarr.tok_idx, lcp=sufarr.lcp), 'models/{}_sufarr.joblib'.format(args.model_name))
