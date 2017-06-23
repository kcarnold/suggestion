# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:13:45 2017

@author: kcarnold
"""

import numpy as np
from suggestion.lang_model import Model
from suggestion.paths import paths
#%%
lowercase = Model.from_basename('yelp_train', paths.model_basename('yelp_train'))
truecase = Model.from_basename('yelp_train_truecase', paths.model_basename('yelp_train_truecase'))
#%%
import cytoolz
#%%
case_options = cytoolz.groupby(lambda x: x.lower(), truecase.id2str)
#%%
def infer_true_case(sent_toks):
    state = truecase.get_state(["<S>"], bos=True)[0]
    result = []
    for tok in sent_toks:
        options = case_options.get(tok, [tok])
        if len(options) == 1:
            result.append(options[0])
            continue
        vocab_indices = [truecase.model.vocab_index(opt) for opt in options]
        scores = truecase.eval_logprobs_for_words(state, vocab_indices)
        chosen_idx = np.argmax(scores)
        result.append(options[chosen_idx])
        state = truecase.advance_state(state, options[chosen_idx])[0]
    return result
' '.join(infer_true_case("i went here last wednesday night because my friend bob and i wanted some new mediterranean food while we were in new york .".split()))
