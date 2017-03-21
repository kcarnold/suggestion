import joblib
import subprocess

def spacy_tok_to_doc(spacy_toked_str):
    res = []
    for sent_str in spacy_toked_str.lower().split('\n'):
        res.append('<S>')
        res.extend(sent_str.split())
        res.append('</S>')
    return res


def dump_kenlm(model_name, tokenized_sentences):
    # Dump tokenized sents / docs, one per line,
    # to a file that KenLM can read, and build a model with it.
    with open('models/{}.txt'.format(model_name), 'w') as f:
        for toks in tokenized_sentences:
            print(toks, file=f)
    subprocess.run(['./scripts/make_model.sh', model_name])


mem = joblib.Memory('cache', mmap_mode='r')

