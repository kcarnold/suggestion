import tqdm

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


def analyze_sentiment(gcl_client, content):
    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)
    annotations = gcl_client.analyze_sentiment(document=document)
    # for index, sentence in enumerate(annotations.sentences):
    #     sentence_sentiment = sentence.sentiment.score
    #     print('Sentence {} has a sentiment score of {}'.format(
    #         index, sentence_sentiment))
    return dict(
        score=annotations.document_sentiment.score,
        magnitude=annotations.document_sentiment.magnitude)

memoized = {}
def memoize_analyze_sentiment(gcl_client, content):
    if content not in memoized:
        memoized[content] = analyze_sentiment(gcl_client, content)
    return memoized[content]

if __name__ == '__main__':
    import os
    os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', os.path.expanduser('~/.config/interactive text rec-b6e7a469c720.json'))
    gcl_client = language.LanguageServiceClient()

    # samples = json.load(samples, open(paths.parent / 'gruntwork' / f'comparisons_multisys_existing_reviews_5words.json', 'r'))

    results = []
    for sample in tqdm.tqdm(samples):
        sample = sample.copy()
        systems = dict(sample.pop('sugg'), true=sample['true_follows'])
        for system_name, sugg in systems.items():
            content = sample['context'] + ' ' + sugg
            sentiment = memoize_analyze_sentiment(gcl_client, content)
            # print(system_name, content, sentiment['score'])
            results.append(dict(sample, system_name=system_name, **sentiment))
