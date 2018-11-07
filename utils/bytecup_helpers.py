import pandas as pd
import ipdb
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tokenize import word_tokenize


def process_sentence(sentence):
    words = word_tokenize(sentence)
    words = [x.lower() for x in words]
    words = [y for x in words for y in x.split('-')]
    words = [y for x in words for y in x.split('.')]
    return words

def process_sentence_for_body(sentence):
    words = process_sentence(sentence)
    lmtzr = WordNetLemmatizer()
    words = [lmtzr.lemmatize(word) for word in words if word]
    return words

def read_all_names():
    name_txt = '/Users/pjs/byte_play/data/general/names.txt'
    names = []
    with open(name_txt, 'r') as f:
        for i, line in enumerate(f):
            if i == 5000:
                break
            name = line.split(' ')[0]
            names.append(name)

    names = [x.lower() for x in names]
    return set(names)