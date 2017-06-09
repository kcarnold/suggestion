import nltk
from suggestion import suggestion_generator

#%%
example_doc = open('example_doc.txt').read()
words = nltk.word_tokenize(example_doc)
#%%
LENGTH = 30
conds = dict(
    A=dict(rare_word_bonus=1.0, useSufarr=True, temperature=0.),
    B=dict(rare_word_bonus=0.0, useSufarr=True, temperature=0.)
    )

model = get_model('yelp_train')
for i in range(0, 50):
    prefix = ' '.join(words[:i])
    print()
    print(prefix[-50:])

    for name, cond in conds.items():
        phrases = suggestion_generator.get_suggestions(
                    prefix + ' ', [],
                    domain='yelp_train',
                    rare_word_bonus=cond.get('rare_word_bonus', 1.0),
                    use_sufarr=cond.get('useSufarr', False),
                    length=LENGTH)
        print(' {:8}: {}'.format(name, ' '.join(phrases[0][0])))
