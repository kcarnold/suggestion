import random
random.seed(0)

personality_data = {
    "NFC": [[
        "Like to solve complex problems.", "Need things explained only once.",
        "Can handle a lot of information.",
        "Love to think up new ways of doing things.",
        "Am quick to understand things.", "Love to read challenging material."
    ], [
        "Have difficulty understanding abstract ideas.",
        "Try to avoid complex people.", "Avoid difficult reading material.",
        "Avoid philosophical discussions."
    ]],
    "Openness": [[
        "Believe in the importance of art.",
        "Have a vivid imagination.",
        "Tend to vote for liberal political candidates.",
        "Carry the conversation to a higher level.",
        "Enjoy hearing new ideas.",
    ], [
        "Am not interested in abstract ideas.",
        "Do not like art.",
        "Avoid philosophical discussions.",
        "Do not enjoy going to art museums.",
        "Tend to vote for conservative political candidates.",
    ]],
    "Extraversion": [[
        "Feel comfortable around people.",
        "Make friends easily.",
        "Am skilled in handling social situations.",
        "Am the life of the party.",
        "Know how to captivate people.",
    ], [
        "Have little to say.",
        "Keep in the background.",
        "Would describe my experiences as somewhat dull.",
        "Don't like to draw attention to myself.",
        "Don't talk a lot.",
    ]],
    "Neuroticism": [[
        "Often feel blue.",
        "Dislike myself.",
        "Am often down in the dumps.",
        "Have frequent mood swings.",
        "Panic easily.",
    ], [
        "Rarely get irritated.",
        "Seldom feel blue.",
        "Feel comfortable with myself.",
        "Am not easily bothered by things.",
        "Am very pleased with myself. ",
    ]]
}

traits_to_use = [
    "Openness",
    "NFC",
    "Neuroticism",
    "Extraversion",
]


def grammaticalize(item):
    return "I " + item[0].lower() + item[1:]


def gen_inventory(traits_to_use):
    items = {
        (trait, pn_idx, item_idx, grammaticalize(item))
        for trait in traits_to_use
        for pn_idx, posneg in enumerate(personality_data[trait])
        for item_idx, item in enumerate(posneg)
    }
    # Each batch should have:
    #  an even division between traits
    #  for each trait, an even division of positive and negative
    return sorted(items, key=lambda x: (x[2], x[1], random.random()))




if __name__ == '__main__':
    items = gen_inventory(traits_to_use)
    # print('\n'.join('|'.join(str(x) for x in item) for item in items))
    print('\n'.join(item[-1] for item in items))
