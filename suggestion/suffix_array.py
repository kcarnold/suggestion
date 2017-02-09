import numpy as np


class DocSuffixArray:
    def __init__(self, docs, suffix_array, doc_idx, tok_idx, lcp):
        self.docs = docs
        self.suffix_array = suffix_array
        self.doc_idx = doc_idx
        self.tok_idx = tok_idx
        self.lcp = lcp

    @classmethod
    def construct(cls, docs, max_toks_per_doc):
        print("Prepare suffix array")
        num_sa_toks = sum(min(max_toks_per_doc, len(doc)) for doc in docs)
        sa_doc_idx = np.empty(num_sa_toks, dtype=int)
        sa_tok_idx = np.empty(num_sa_toks, dtype=int)
        ptr = 0
        master_token_list = []
        for doc_idx, doc in enumerate(docs):
            tok_list = [tok.lower() for tok in doc[:max_toks_per_doc]]
            sa_doc_idx[ptr:ptr+len(tok_list)] = doc_idx
            sa_tok_idx[ptr:ptr+len(tok_list)] = np.arange(len(tok_list))
            master_token_list.extend(tok_list)
            ptr += len(tok_list)

        print("Build suffix array")
        suffix_array = np.array(build_suffix_array(master_token_list, max_toks_per_doc))

        print("Reindex data structures")
        doc_idx = sa_doc_idx[suffix_array]
        tok_idx = sa_tok_idx[suffix_array]

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





# Adapted from http://codereview.stackexchange.com/questions/87335/suffix-array-construction-in-on-log2-n

from itertools import chain, islice

def build_suffix_array(A, max_length):
    """Return a list of the starting positions of the suffixes of the
    sequence A in sorted order.

    For example, the suffixes of ABAC, in sorted order, are ABAC, AC,
    BAC and C, starting at positions 0, 2, 1, and 3 respectively:

    >>> suffix_array('ABAC')
    [0, 2, 1, 3]

    """
    # This implements the algorithm of Vladu and Negruşeri; see
    # http://web.stanford.edu/class/cs97si/suffix-array.pdf

    L = sorted((a, i) for i, a in enumerate(A))
    n = len(A)
    count = 1
    while count < min(n, max_length):
        # Invariant: L is now a list of pairs such that L[i][1] is the
        # starting position in A of the i'th substring of length
        # 'count' in sorted order. (Where we imagine A to be extended
        # with dummy elements as necessary.)

        P = [0] * n
        for (r, i), (s, j) in zip(L, islice(L, 1, None)):
            P[j] = P[i] + (r != s)

        # Invariant: P[i] is now the position of A[i:i+count] in the
        # sorted list of unique substrings of A of length 'count'.

        L = sorted(chain((((P[i],  P[i+count]), i) for i in range(n - count)),
                         (((P[i], -1), i) for i in range(n - count, n))))
        count *= 2
    return [i for _, i in L]


def bisect_left(less_than, hi):
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if less_than(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo
