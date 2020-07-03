import os
import pandas as pd
import spacy
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.tokenize import sent_tokenize


class Cranfield(object):

    def __init__(self, folder, textfields=['text']):
        self.textfields = textfields
        self.folder = folder
        self.docfile = os.sep.join([self.folder,
                                    'cran.all.1400'])
        self.queryfile = os.sep.join([self.folder,
                                      'cran.qry'])
        self.relevancefile = os.sep.join([self.folder,
                                          'cranqrel'])

    def readdocs(self, file=None):
        if file is None:
            file = self.docfile
        with open(file, 'r') as infile:
            lines = infile.readlines()
        docs, doc, current = [], None, None
        for line in lines:
            if line.startswith('.I'):
                if doc is None:
                    doc = {}
                else:
                    yield doc
                    doc = {}
                doc['id'] = int(line.rstrip().split()[-1])
            elif line.startswith('.T'):
                doc['title'] = ""
                current = 'title'
            elif line.startswith('.A'):
                doc['author'] = ""
                current = 'author'
            elif line.startswith('.B'):
                doc['venue'] = ""
                current = 'venue'
            elif line.startswith('.W'):
                doc['text'] = ""
                current = 'text'
            else:
                doc[current] += " " + line.strip()
        yield doc

    def __iter__(self):
        for doc in self.readdocs():
            text = " ".join([doc[field] for
                             field in self.textfields])
            yield (doc['id'], text)

    def count(self):
        return len(list(self.readdocs()))

    def relevance(self):
        data = {'query': [], 'doc': [], 'relevance': []}
        with open(self.relevancefile, 'r') as rf:
            lines = rf.readlines()
        for line in lines:
            numbers = [int(x) for x in line.strip().split()]
            data['query'].append(numbers[0])
            data['doc'].append(numbers[1])
            data['relevance'].append(numbers[2])
        return pd.DataFrame(data)


class Tokenizer(object):

    def __init__(self, language):
        try:
            self.iso, self.extended = language
        except:
            print('Error: language should be provided in the form (iso-2, extended)')
        self.stemmer = SnowballStemmer(self.extended)
        self.nlp = spacy.load(self.iso)
        self.keys = ['document', 'sentence', 'position', 'text', 'lower', 'lemma',
                     'pos', 'tag', 'dep',
                     'shape', 'is_alpha', 'is_stop', 'stem']

    def tokenize(self, text_id, text, drop_apostrophe=False):
        if drop_apostrophe:
            text = text.replace("'", " ")
        tokens = []
        for j, sentence in enumerate(sent_tokenize(text)):
            doc = self.nlp(sentence.strip())
            for i, token in enumerate(doc):
                lower = token.text.lower()
                tag_data = [tuple(x.split('=')) for x in
                            token.tag_.split('|')]
                try:
                    tag = dict(tag_data)
                except ValueError:
                    tag = tag_data[0][0]
                data = [text_id, j, i, token.text, lower, token.lemma_,
                        token.pos_, tag, token.dep_,
                        token.shape_, token.is_alpha, token.is_stop, self.stemmer.stem(lower)]
                tokens.append(dict(zip(self.keys, data)))
        return tokens
