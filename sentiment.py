import csv
from nltk.corpus import wordnet as wn
import numpy as np
from nltk.wsd import lesk

file = open("sentiDict.txt", newline="\n")
filereader = csv.reader(file, delimiter=",")
next(filereader)

sentiDict = {}

for row in filereader:
    sentiDict[(row[0], row[1])] = (float(row[2]), float(row[3]))


def getSent(word):
    # synset = lesk(sent.text, word.text)
    # print(synset)
    # if synset.pos() not in ["a", "s"]:
    synset = None
    for set in wn.synsets(word):
        if set.pos() in ["a", "s"]:
            synset = set
            # print("New synset ->" + str(synset))
            break
    if synset is None:
        print("None SET for: "+word)
        return None
    # print("iD: " + str(synset.offset()))
    score = sentiDict[("a" if synset.pos() == "s" else synset.pos(), str(synset.offset()).zfill(8))]
    return score[0] - score[1]

# getSent("Good selection of classes of beers and mains.", "good")
# synsets = wn.synsets("dirty")
# print(synsets)
# for s in synsets:
#     print(s, s.offset())