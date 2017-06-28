import joblib
import subprocess


def dump_kenlm(model_name, tokenized_sentences, **model_args):
    # Dump tokenized sents / docs, one per line,
    # to a file that KenLM can read, and build a model with it.
    with open('models/{}.txt'.format(model_name), 'w') as f:
        for toks in tokenized_sentences:
            print(toks, file=f)
    make_model(model_name, **model_args)


def make_model(model_name, order=5, prune=2):
    from suggestion.paths import paths
    model_full_name = str(paths.models / model_name)
    kenlm_bin = paths.parent.parent / 'kenlm' / 'build' / 'bin'
    lmplz_args = ['-o', str(order)]
    if prune is not None:
        lmplz_args.append('--prune')
        lmplz_args.append(str(prune))
    lmplz_args.append('--verbose_header')
    with open(model_full_name + '.txt', 'rb') as in_file, open(model_full_name + '.arpa', 'wb') as arpa_file:
        subprocess.run([str(kenlm_bin / 'lmplz')] + lmplz_args, stdin=in_file, stdout=arpa_file)
    subprocess.run([str(kenlm_bin / 'build_binary'), model_full_name + '.arpa', model_full_name + '.kenlm'])


def flatten_dict(x, prefix=''):
    result = {}
    for k, v in x.items():
        if isinstance(v, dict):
            result.update(flatten_dict(v, prefix=prefix + k + '_'))
        else:
            result[prefix + k] = v
    return result



mem = joblib.Memory('cache', mmap_mode='r')
