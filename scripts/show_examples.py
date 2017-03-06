import nltk
from suggestion import suggestion_generator

example_doc = open('example_doc.txt').read()
words = nltk.word_tokenize(example_doc)

conds = dict(
    rare=dict(rare_word_bonus=5.0, useSufarr=True, temperature=0.),
    common=dict(rare_word_bonus=0.0, useSufarr=True, temperature=0.)
    )

for i in range(10, 30):
    prefix = ' '.join(words[:i])
    print()
    print(prefix[-25:])
    for name, cond in conds.items():
        phrases = suggestion_generator.get_suggestions(
                    prefix + ' ', [],
                    domain='yelp_train',
                    rare_word_bonus=cond.get('rare_word_bonus', 1.0),
                    use_sufarr=cond.get('useSufarr', False),
                    temperature=cond.get('temperature', 0.),
                    length=15)
        print(' {:8}: {}'.format(name, ' '.join(phrases[0][0])))

