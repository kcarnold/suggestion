import argparse
import pickle
from sys import intern
import tqdm
import joblib
from suggestion.util import spacy_tok_to_doc, dump_kenlm
from suggestion.suffix_array import DocSuffixArray
from suggestion import tokenization
import pandas as pd
import numpy as np

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

def convert_tokenization(doc):
    doc = doc.lower()
    doc = sub_numbers(doc)
    sents = doc.split('\n')
    sents = [cant_type.sub('', sent) for sent in sents]
    return [intern(word) for word in spacy_tok_to_doc(sents)]

def typeable_chars(text):
    text = tokenization.URL_RE.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    return cant_type.sub('', sub_numbers(text.lower()))

def tokenize(text):
    # NOTE: Yelp preprocessing runs this same fn (almost) in preprocess_yelp_v2, copy-pasted.
    import nltk
    text = text.replace("Mr.", "Mr").replace("Mrs.", "Mrs").replace("Ms.", "Ms")
    text = tokenization.URL_RE.sub(" ", text)
    sents = nltk.sent_tokenize(text)
    token_spaced_sents = (' '.join(sent[a:b] for a, b in tokenization.token_spans(sent)) for sent in sents)
    return '\n'.join(token_spaced_sents)


def preprocess_csv(input_filename, model_name):
    import pandas as pd
    data = pd.read_csv(input_filename)
    dump_kenlm(model_name, (' '.join(convert_tokenization(tokenize(text))) for text in tqdm.tqdm(data.Text)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',
                        help='Input data filename (pickle)',
                        default='yelp_preproc/train_data.pkl')
    parser.add_argument('--model-name', default="yelp_train",
                        help="model name")
    parser.add_argument('--subsample-stars', action='store_true',
        help="Breakout the model by review star ratings.")
    args = parser.parse_args()

    print("Loading...", flush=True)
    data = pd.read_pickle(args.input)
    reviews = data['data']

    tokenized_reviews = [convert_tokenization(tokenized)
        for tokenized in tqdm.tqdm(reviews.tokenized, desc="Converting format")]

    if args.subsample_stars:
        positive_reviews = np.flatnonzero(reviews.stars_review > 3)
        negative_reviews = np.flatnonzero(reviews.stars_review < 3)
        sample_size = min(len(positive_reviews), len(negative_reviews))
        positive_reviews_selected = np.random.choice(positive_reviews, size=sample_size, replace=False)
        negative_reviews_selected = np.random.choice(negative_reviews, size=sample_size, replace=False)
        dump_kenlm(f"{args.model_name}-5star", (' '.join(tokenized_reviews[idx]) for idx in tqdm.tqdm(positive_reviews_selected, desc="Writing pos")))
        dump_kenlm(f"{args.model_name}-1star", (' '.join(tokenized_reviews[idx]) for idx in tqdm.tqdm(negative_reviews_selected, desc="Writing neg")))
        by_star_rating_sample_size = min(reviews.stars_review.value_counts())
        balanced_indices = np.concatenate(
            [np.random.choice(np.flatnonzero(reviews.stars_review == stars), size=by_star_rating_sample_size, replace=False)
            for stars in [1,2,3,4,5]], axis=0)
        np.random.shuffle(balanced_indices) # for good measure, even tho it shouldn't matter.
        dump_kenlm(f"{args.model_name}-balanced", (' '.join(tokenized_reviews[idx]) for idx in tqdm.tqdm(balanced_indices, desc="Writing balanced")))
    else:
        print("Saving reviews")
        with open('models/tokenized_reviews.pkl', 'wb') as f:
            pickle.dump(tokenized_reviews, f, -1)

        dump_kenlm(args.model_name, (' '.join(doc) for doc in tqdm.tqdm(tokenized_reviews, desc="Writing")))

        sufarr = DocSuffixArray.construct(tokenized_reviews)
        joblib.dump(dict(suffix_array=sufarr.suffix_array, doc_idx=sufarr.doc_idx, tok_idx=sufarr.tok_idx, lcp=sufarr.lcp), 'models/{}_sufarr.joblib'.format(args.model_name))

        # print("Training char model")
        # dump_kenlm(args.model_name+"_char", (' '.join(typeable_chars(text).replace(' ', '_')) for text in reviews.text), script='./scripts/make_char_model.sh')
