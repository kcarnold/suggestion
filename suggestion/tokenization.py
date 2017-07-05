import re
import string
from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer

WORD_RE = re.compile(r'\w+(?:[\',:]\w+)*')
END_PUNCT = set('.,?!:')
END_PUNCT_OR_SPACE = set('.,?!: ')
URL_RE = re.compile(r'\bhttp[s]?://[^\s]*[^\s.]')


def token_spans(text):
    for match in re.finditer(r'[^-/\s]+', text):
        start, end = match.span()
        token_match = WORD_RE.search(text, start, end)
        if token_match is not None:
            span = token_match.span()
            yield span
            tok_end = span[1]
            if tok_end < end and text[tok_end] in END_PUNCT:
                yield tok_end, tok_end + 1


START_DOC = '<D>'
START_PARA = '<P>'
START_SENT = '<S>'
END_SENT = '</S>'

paragraph_re = re.compile(r'(?:[ ]*[^\s][^\n]*[\n]?)+')
paragraph_tokenizer = RegexpTokenizer(paragraph_re)
sentence_tokenizer = PunktSentenceTokenizer()

def tokenize(doc):
    res = [START_DOC]
    afters = []
    end_of_prev_para = 0
    for para_start, para_end in paragraph_tokenizer.span_tokenize(doc):
        afters.append(doc[end_of_prev_para:para_start])
        para = doc[para_start:para_end]
        end_of_prev_para = para_end
        end_of_prev_sentence = 0
        res.append(START_PARA)
        for sent_start, sent_end in sentence_tokenizer.span_tokenize(para):
            sent = para[sent_start:sent_end]
            tspans = list(token_spans(sent))
            if not tspans:
                assert all(c == ' ' or c in END_PUNCT for c in sent)
                continue
            afters.append(para[end_of_prev_sentence:sent_start])
            end_of_prev_sentence = sent_end
            res.append(START_SENT)
            end_of_prev_token = 0
            for tok_start, tok_end in tspans:
                afters.append(sent[end_of_prev_token:tok_start])
                res.append(sent[tok_start:tok_end])
                end_of_prev_token = tok_end
            res.append(END_SENT)
            afters.append(sent[end_of_prev_token:])
        end_of_prev_para -= len(para) - end_of_prev_sentence
    afters.append(doc[end_of_prev_para:])
    return res, afters


def tokenize_mid_document(doc_so_far):
    if len(doc_so_far.strip()) == 0:
        return [START_DOC, START_PARA, START_SENT, ''], ['', '', '', '']
    tok_list, afters = tokenize(doc_so_far)
    if doc_so_far.endswith('\n\n'):
        # starting a new paragraph
        if tok_list[-1] in [START_PARA, START_SENT]:
            print("Huh? Ended with double-newlines but also with start-of-para?", repr(tok_list[-5:]))
        tok_list += [START_PARA, START_SENT, '']
        afters += ['', '', '']
    else:
        if tok_list[-1] != END_SENT:
            assert set(doc_so_far) <= END_PUNCT_OR_SPACE
        if tok_list[-2] in '.?!':
            # Real EOS
            tok_list += [START_SENT, '']
            afters += ['', '']
        elif doc_so_far[-1] in string.whitespace:
            # The last EOS was spurious, but we're not mid-word.
            tok_list[-1] = ""
        else:
            # The last EOS was spurious, but we ARE mid-word.
            tok_list.pop(-1)
            after = afters.pop(-1)
            afters[-1] += after

    return tok_list, afters


def sentence_per_line_tokenize(text):
    res = []
    for sent in sentence_tokenizer.tokenize(text):
        toks = ' '.join(sent[tok_start:tok_end] for tok_start, tok_end in token_spans(sent))
        if toks:
            res.append(toks)
    return '\n'.join(res)
