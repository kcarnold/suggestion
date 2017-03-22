import joblib
import subprocess

def spacy_tok_to_doc(spacy_sent_strs):
    res = []
    for i, sent_str in enumerate(spacy_sent_strs):
        res.append('<S>' if i > 0 else '<D>')
        res.extend(sent_str.lower().split())
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

