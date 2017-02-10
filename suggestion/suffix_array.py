import numpy as np
from suggestion.suffix_sort import suffix_sort

class DocSuffixArray:
    def __init__(self, docs, suffix_array, doc_idx, tok_idx, lcp):
        self.docs = docs
        self.suffix_array = suffix_array
        self.doc_idx = doc_idx
        self.tok_idx = tok_idx
        self.lcp = lcp

    @classmethod
    def construct(cls, docs):
        print("Prepare suffix array")
        num_sa_toks = sum(len(doc) for doc in docs)
        sa_doc_idx = np.empty(num_sa_toks, dtype=int)
        sa_tok_idx = np.empty(num_sa_toks, dtype=int)
        ptr = 0
        master_token_list = []
        for doc_idx, doc in enumerate(docs):
            tok_list = [tok.lower() for tok in doc]
            sa_doc_idx[ptr:ptr+len(tok_list)] = doc_idx
            sa_tok_idx[ptr:ptr+len(tok_list)] = np.arange(len(tok_list))
            master_token_list.extend(tok_list)
            ptr += len(tok_list)

        print("Building vocabulary")
        assert len(master_token_list) == num_sa_toks
        # FIXME: sort end-of-document at the end. (or beginning??)
        vocab = sorted(set(master_token_list))
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        print("Mapping vocabulary")
        tok_array = np.empty(num_sa_toks, dtype=np.int32)
        for i, tok in enumerate(master_token_list):
            tok_array[i] = word2idx[tok]

        print("Build suffix array")
        suffix_array, sufarr_inverse = suffix_sort(tok_array, x_min=0, x_max=len(vocab))

        print("Reindex data structures")
        doc_idx = sa_doc_idx[suffix_array[1:]]
        tok_idx = sa_tok_idx[suffix_array[1:]]

        print("Build LCP array")
        N = len(sa_doc_idx)
        lcp = lcp = np.zeros(N - 1, dtype=int)
        suf = docs[doc_idx[0]][tok_idx[0]:]
        for i in range(N - 1):
            next_suf = docs[doc_idx[i + 1]][tok_idx[i + 1]:]
            m = min(len(suf), len(next_suf))
            for j in range(m):
                if suf[j] != next_suf[j]:
                    lcp[i] = j
                    break
            else:
                lcp[i] = m
            suf = next_suf
        return cls(docs=docs, suffix_array=suffix_array, doc_idx=doc_idx, tok_idx=tok_idx, lcp=lcp)

    def get_suffix_by_idx(self, idx):
        return self.docs[self.doc_idx[idx]][self.tok_idx[idx]:]

    def search_range(self, prefix_lowered):
        docs = self.docs
        doc_idx = self.doc_idx
        tok_idx = self.tok_idx
        def suf_less_than(x):
            return lambda idx: tuple(docs[doc_idx[idx]][tok_idx[idx]:]) < x
        N = len(doc_idx)
        return (
            bisect_left(suf_less_than(prefix_lowered), N),
            bisect_left(suf_less_than(prefix_lowered[:-1] + (prefix_lowered[-1] + '\uffff',)), N))


def bisect_left(less_than, hi):
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if less_than(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo
