#%%
import itertools
import tqdm
#%%
def analyze_requests_from_parsed_logfile(requests):
#%%
#%lprun -f suggestion_generator.beam_search_phrases -f suggestion_generator.get_suggestions
    responses = [(request,) + do_request(request) for request in requests[:10]]
    #%%
    import pickle
    pickle.dump(responses, open('responses_2017-03-14-15-12.pkl','wb'), -1)
    #%%
    # Find the longest queries.
    durs = [dur for req, dur, resp in responses]
    [requests[i] for i in np.argsort(durs)[-10:]]
    #%%
    model = suggestion_generator.get_model('yelp_train')
    unigram_probs = suggestion_generator.get_unigram_probs(model)
    #%%
    #responses_by_page = [(page, list(responses)) for page, responses in itertools.groupby(responses, lambda response: response[0]['page'])]
    df = []
    for request, dur, phrases in responses:
        if phrases is None or phrases == []: continue
        start_at = 1 if request['cur_word'] else 0
        page, block = request['page'].split('-', 1)
        block = int(block)
        ent = dict(page=page, block=block, dur=dur)
        indices = [[model.model.vocab_index(word) for word in phrase] for phrase, probs in phrases]
        indices_flat = np.array([idx for phrase_indices in indices for idx in phrase_indices[start_at:] if idx])
        ent['mlf'] = np.mean(unigram_probs[indices_flat])
        df.append(ent)
    df = pd.DataFrame(df)

    #sns.distplot(df.dur, groupby=df.block)#.groupby('block').dur.plot(legend='best')
    sns.violinplot(x='mlf', y='page', hue='block', data=df.query('mlf > -10'), split=True)
    #responses_by_page = [(page, [do_request(request) for request in requests]) for page, requests in tqdm.tqdm(requests_by_page)]

#%%
if False:
#%%
    analyzed = dict(json.load(open('frontend/analyzed.json')))
    requests = [r for r in analyzed['cf35a8']['requests'] if True]#not r['cur_word']]
