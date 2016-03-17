from pickle import load
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import numpy as np
import nltk, settings

OPINION = 1
NEWS = 0
TRAIN = 1
TEST = 0
STOP_WORDS = stopwords.words('english')

with open("shared_words.txt", "r") as f:
    SHARED = f.read()
    SHARED = set(SHARED.split())

# LR classifier

def get_words_from_string(article):
    return [w.lower() for w in nltk.word_tokenize(article) if w.isalpha()]

def ambiguous_words(article):
    return len([w for w in article if w in SHARED])/len(article)

def stopwords_proportion(article):
    return len([w for w in article if w in STOP_WORDS])/len(article)

def gen_word_tag_pairs(article):
    tags = nltk.pos_tag(article)
    return [word+"_"+tag for word, tag in tags]

def build_opinion_seedwords():
    seedset = []
    seedroots = ['good', 'aseome','beautiful','decent','nice', 'excellent','good','bad','expensive','faulty','horrible',
               'poor','stupid','cheap','decent','effective','fantastic','happy','impress','jittery','light', 'madly',
                'nice', 'outstanding', 'perfect', 'quick', 'responsive', 'sharp', 'terrible','ultimate', 'wonderful']
    for word in seedroots:
        seedset += [s.name().split('.')[0] for s in wn.synsets(word)]
    return set(seedset)

SEEDWORDS = build_opinion_seedwords()

def seedword_proportion(article):
    return len([w for w in article if w in SEEDWORDS])/len(article)

def vectorize(vectorizer, list_of_texts):
    compressed_vectors = vectorizer.transform(list_of_texts)
    return compressed_vectors.toarray()

def prepare_clf():
    input_pkl = open('model_lr.pkl', 'rb')
    settings.clf = load(input_pkl)
    input_pkl.close()
    input_pkl = open('vectorizer.pkl', 'rb')
    settings.vectorizer = load(input_pkl)
    input_pkl.close()
    input_pkl = open('scaler.pkl', 'rb')
    settings.scaler = load(input_pkl)
    input_pkl.close()

def add_features_from_article(features, article):
    p_shared = np.array([[ambiguous_words(article)]])
    p_stopwords = np.array([[stopwords_proportion(article)]])
    p_seedwords = np.array([[seedword_proportion(article)]])
    return np.append(np.append(np.append(features, p_shared, axis=1), p_stopwords, axis=1), p_seedwords, axis=1)

def predict_sample(article, vectorizer, clf, scaler):
    article = get_words_from_string(article)
    sample = vectorize(vectorizer, [' '.join(gen_word_tag_pairs(article))])
    sample = add_features_from_article(sample, article)
    sample = scaler.transform(sample)
    y_pred = clf.predict(sample)
    return y_pred