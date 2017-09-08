import os
try:
    import ujson as json
except ImportError:
    import json
import re
import numpy as np
from suggestion.util import mem
from suggestion.paths import paths
import subprocess

#
# Data for decoding surveys.
#

skip_col_re = re.compile(
    r'Great.job|Q_\w+|nextURL|clientId|Timing.*|Browser.*|Location.*|Recipient.*|Response.+|ExternalDataReference|Finished|Status|IPAddress|StartDate|EndDate|Welcome.+|Display Order|Demographic Questions|Closing Survey.+|revisionDesc|prewrite')

prefix_subs = {
    "How much do you agree with the following statements about the suggestions that the system gave?-They ": "suggs-",
    "How much do you agree with the following statements?-The suggestions ": "suggs-",
    "How much do you agree with the following statements about the words or phrases that the keyboard...-They ": "suggs-",
    "Now think about the brainstorming you did before the final writing. How much do you agree with th...-": "brainstorm-",
    "Think about when you were typing out your ${e://Field/revisionDesc}. How much do you agree with t...-": "final-",
    "How Accurately Can You Describe Yourself? Describe yourself as you generally are now, not as you...-": "pers-",
    "Describe yourself as you generally are now, not as you wish to be in the future. Describe yoursel...-": "pers-",
}

decode_scales = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Somewhat disagree": 3,
        "Neither agree nor disagree": 4,
        "Somewhat agree": 5,
        "Agree": 6,
        "Strongly agree": 7,

        "Very Inaccurate": 1,
        "Moderately Inaccurate": 2,
        "Neither Accurate Nor Inaccurate": 3,
        "Moderately Accurate": 4,
        "Very Accurate": 5}


def get_rev(logpath):
    with open(logpath) as logfile:
        for line in logfile:
            line = json.loads(line)
            if 'rev' in line:
                return line['rev']

def checkout_old_code(git_rev):
    import shutil
    by_rev = paths.parent / 'old-code'
    rev_root = by_rev / git_rev
    if not os.path.isdir(rev_root):
        print("Checking out repository at", git_rev)
        subprocess.check_call(['git', 'clone', '..', git_rev], cwd=by_rev)
        subprocess.check_call(['git', 'checkout', git_rev], cwd=rev_root)
        print("Installing npm packages")
        subprocess.check_call(['yarn'], cwd=os.path.join(rev_root, 'frontend'))


@mem.cache
def get_log_analysis_raw(logpath, logfile_size, git_rev=None, analysis_files=None):
    # Ignore analysis_files; just use them to know when to invalidate the cache.
    checkout_old_code(git_rev)
    analyzer_path = os.path.join(paths.parent, 'frontend', 'analysis')
    with open(logpath) as logfile:
        result = subprocess.check_output([analyzer_path], stdin=logfile)
        assert len(result) > 0
        return result


def get_log_analysis(participant, git_rev=None):
    analysis_files = {
        name: open(paths.parent / 'frontend' / name).read()
        for name in ['analyze.js', 'analysis', 'src/Analyzer.js']
    }
    logpath = paths.parent / 'logs' / (participant+'.jsonl')
    if git_rev is None:
        git_rev = get_rev(logpath)
    logfile_size = os.path.getsize(logpath)

    result = get_log_analysis_raw(logpath, logfile_size, git_rev=git_rev, analysis_files=analysis_files)
    analyzed = json.loads(result)
    analyzed['git_rev'] = git_rev
    return analyzed



def classify_annotated_event(evt):
    typ = evt['type']
    if typ in {'externalAction', 'next', 'resized', 'tapText'}:
        return None
    text = evt['curText']
    null_word = len(text) == 0 or text[-1] == ' '
    text = text.strip()
    bos = len(text) == 0 or text[-1] in '.?!'
    if typ == 'tapKey':
        return 'tapKey'
    if typ == 'tapBackspace':
        return 'tapBackspace'
    if typ == 'tapSuggestion':
        if bos:
            sugg_mode = 'bos'
        elif null_word:
            sugg_mode = 'full'
        else:
            sugg_mode = 'part'
        return 'tapSugg_' + sugg_mode
    assert False, typ



def get_content_stats_single_suggestion(sugg, word_freq_analyzer):
    from suggestion import suggestion_generator
    sugg = sugg.copy()
    meta = sugg.pop('flags')

    if not meta['domain'].startswith('yelp'):
        return

    if sugg['cur_word']:
        # Skip partial words.
        return

    model = suggestion_generator.Model.get_or_load_model(meta['domain'])
    try:
        toks = suggestion_generator.tokenize_sofar(sugg['sofar'])
    except:
        # Tokenization failed.
        return
    # Optimization: trim context to the n-gram level, plus some padding.
    toks = toks[-10:]
    state = model.get_state(toks)[0]
    clf_startstate = suggestion_generator.sentiment_classifier.get_state(toks)
    res = []
    for sugg_slot, rec in enumerate(sugg['recs']['predictions']):
        phrase = rec['words']
        if phrase:
            sentiment_posteriors = suggestion_generator.sentiment_classifier.classify_seq_by_tok(clf_startstate, phrase)
            sentiment = np.mean(sentiment_posteriors, axis=0) @ suggestion_generator.sentiment_classifier.sentiment_weights
        else:
            sentiment = None
        analyzer_indices = [word_freq_analyzer.word2idx.get(tok) for tok in phrase]
        res.append(dict(
            request_id=sugg['request_id'],
            sugg_slot=sugg_slot,
            sugg_contextual_llk=model.score_seq(state, phrase)[0],
            sugg_unigram_llk=np.nanmean(np.array([word_freq_analyzer.log_freqs[idx] if idx is not None else np.nan for idx in analyzer_indices])),
            sugg_sentiment=sentiment))
    return res

