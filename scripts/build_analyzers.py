from suggestion import analyzers
from suggestion.paths import paths
import pickle

reviews = analyzers.load_reviews()
wordpair_analyzer = analyzers.WordPairAnalyzer.build(reviews)
with open(paths.models / 'wordpair_analyzer.pkl', 'wb') as f:
    pickle.dump(wordpair_analyzer, f, -1)
