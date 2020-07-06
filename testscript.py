import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import fasttext.util
import fasttext
from nltk.corpus import wordnet as wn
import spacy

best = 0
for set in wn.synsets("pasta"):
    for set2 in wn.synsets("food"):
        if wn.wup_similarity(set, set2) > best:
            best = wn.wup_similarity(set, set2)
            sets = (set, set2)

import csv

file = open("sentiDict.txt", newline="\n")
filereader = csv.reader(file, delimiter=",")
next(filereader)
sentiDict = {}

for row in filereader:
    sentiDict[(row[0], row[1])] = (float(row[2]), float(row[3]))


def getSent(word):
    score = 0
    synsets = wn.synsets(word)
    if len(synsets) == 0:
        return None
    for set in synsets:
        if set.pos() in ["a", "s"]:
            synset_scores = sentiDict[("a" if set.pos() == "s" else set.pos(), str(set.offset()).zfill(8))]
            score += synset_scores[0] if synset_scores[0] > synset_scores[1] else -synset_scores[1]

    return score / len(synsets)


print(getSent("terrific"))

analyzer = SentimentIntensityAnalyzer()
word = "tasty"
word2 = "good"
print(analyzer.polarity_scores(word))
print(analyzer.polarity_scores(word2))

