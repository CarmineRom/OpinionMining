import spacy
from spacy.matcher import Matcher
import csv
from nltk.corpus import wordnet as wn
import re
import vaderSentiment.vaderSentiment as vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load english model
print("Loading english model...")
nlp = spacy.load("en_core_web_lg")

# Loading sentimentDict
print("Loading sentiments Dictionary")
print("")
file = open("sentiDict.txt", newline="\n")
filereader = csv.reader(file, delimiter=",")
next(filereader)

sentiDict = {}

for row in filereader:
    sentiDict[(row[0], row[1])] = (float(row[2]), float(row[3]))

analyzer = SentimentIntensityAnalyzer()


def myprint(string):
    words = string.split(" ")
    i = 1
    s = words[0]
    while i < len(words):
        if i % 25 == 0:
            print(s)
            s = words[i]
        else:
            s = s + " " + words[i]
        i += 1
    print(s)


def preprocessChars(sentence):
    rev = re.sub(r"[^a-zA-Z0-9.',:;?]+", ' ', sentence)
    rev = re.sub(r"([;. ]+)([Bb])ut", " but", rev)
    rev = re.sub(r"([;., ]+)([Aa])nd", " and", rev)

    return rev


def pronoun_ref(review, pronoun_token):
    print("IN FUN")
    if pronoun_token.pos_ == "PRON":
        subj_found = False
        i = pronoun_token.i - 1
        while (not subj_found) and i > -1:
            print("IN WHILE ", i)
            if review[i].dep_ == "nsubj" or review[i].dep_ != "pobj" and review[i].pos_ == "NOUN":
                return review[i]
            i -= 1
        if not subj_found:
            return pronoun_token
    else:
        return pronoun_token


def extract_oa_dict(sentence):
    oa_dict = {}

    def addPair(aspect, opinion):
        if aspect not in oa_dict.keys():
            oa_dict[aspect] = [opinion]
        else:
            if opinion[2] not in [a for n, m, a in oa_dict[aspect]]:
                oa_dict[aspect].append(opinion)

    # Find opinion looking for adjectives
    for token in sentence:
        if token.tag_ in ["JJ", "JJS"]:  # JJ-TAG excludes comparative adjectives

            modifier = ""
            negation = False

            for adj_child in token.children:
                if adj_child.dep_ == "advmod":
                    modifier = adj_child.text

            # AMOD-NOUN CASE
            if token.dep_ == "amod" or token.dep_ == "compound":

                for adj_child in token.children:
                    if adj_child.dep_ == "neg":
                        negation = True

                noun = token.head
                if noun.pos_ == "NOUN" or noun.pos_ == "PROPN":

                    # Check if head is a compound noun
                    noun_text = noun.text
                    for noun_child in noun.children:
                        if noun_child.dep_ == "neg":
                            negation = True
                        if noun_child.dep_ == "compound":
                            noun_text = noun_child.text + " " + noun_text

                    if noun.dep_ == "attr" and noun.head.pos_ in ["AUX", "VERB"]:
                        noun_verb = noun.head
                        for verb_child in noun_verb.children:
                            if verb_child.dep_ == "neg":
                                negation = True

                    addPair(noun_text, (negation, modifier, token.text))

                    # Propagate to conj
                    for adj_child in token.children:
                        if adj_child.dep_ == "conj" and adj_child.tag_ in ["JJ", "JJS"]:
                            addPair(noun_text, (negation, modifier, adj_child.text))

                    for noun_child in noun.children:
                        if noun_child.dep_ == "conj" and noun_child.pos_ == "NOUN":
                            addPair(noun_child.text, (negation, modifier, token.text))

            # ACOMP CASE
            if token.dep_ == "acomp" or token.dep_ == "pobj" or token.dep_ == "attr":
                verb = token.head
                while (verb.pos_ != "VERB" and verb.pos_ != "AUX") and verb.dep_ == "conj":
                    verb = verb.head

                for verb_child in verb.children:
                    if verb_child.dep_ == "neg":
                        negation = True

                for verb_child in verb.children:
                    if verb_child.dep_ == "nsubj":
                        subject = verb_child
                        # if verb_child.lower_ in pronouns:
                        #     # Subject is a pronoun, find reference
                        #     subject = pronounRef(review, verb_child)
                        #     print("REFERENCE: ", verb_child, subject)

                        # Check if subject is a compound noun
                        noun_text = subject.text
                        for noun_child in subject.children:
                            if noun_child.dep_ == "compound":
                                noun_text = noun_child.text + " " + noun_text

                        addPair(noun_text, (negation, modifier, token.text))

                        # Propagate to adjective - conj
                        for adj_child in token.children:
                            if adj_child.dep_ == "conj" and adj_child.tag_ in ["JJ", "JJS"]:
                                # print("C"+adj_child.text)
                                # for a in adj_child.subtree:
                                #     print("T"+a.text)
                                for subchild in adj_child.children:
                                    if subchild.dep_ == "neg":
                                        negation = True
                                    if subchild.dep_ == "advmod":
                                        modifier = subchild.text

                                addPair(noun_text, (negation, modifier, adj_child.text))

                        # Propagate to subject - conj
                        for subj_child in subject.children:
                            if subj_child.dep_ == "conj" and subj_child.pos_ == "NOUN":
                                addPair(subj_child.text, (negation, modifier, token.text))

    return oa_dict


# Get Wordnet polarity
def get_sentiW_polarity(word):
    score = 0
    synsets = wn.synsets(word)
    if len(synsets) == 0:
        return 0
    for set in synsets:
        if set.pos() in ["a", "s"]:
            synset_scores = sentiDict[("a" if set.pos() == "s" else set.pos(), str(set.offset()).zfill(8))]
            score += 0 if synset_scores[0] == synset_scores[1] \
                else synset_scores[0] if synset_scores[0] >= synset_scores[1] else -synset_scores[1]

    return score / len(synsets)


def get_polarity(opinion):
    if opinion[2] in ["expensive", "cheap"]:
        polarity = -0.5 if opinion[2] == "expensive" else 0.5
        if opinion[1]:
            if vader.BOOSTER_DICT.get(opinion[1]) is not None:
                polarity += vader.BOOSTER_DICT[opinion[1]] if polarity > 0 else -vader.BOOSTER_DICT[opinion[1]]
        polarity = -polarity if opinion[0] else polarity
        return polarity
    # Vader
    scores = analyzer.polarity_scores(opinion[2])
    if scores["compound"] != 0:
        phrase = ("not " if opinion[0] else "") + opinion[1] + " " + opinion[2]
        scores = analyzer.polarity_scores(phrase)
        return scores["compound"]

    # Wordent Polarity
    else:
        wn_polarity = get_sentiW_polarity(opinion[2])
        if opinion[1] and not -0.1 < wn_polarity < 0.1:
            if vader.BOOSTER_DICT.get(opinion[1]) is not None:
                wn_polarity += vader.BOOSTER_DICT[opinion[1]] if wn_polarity > 0 else -vader.BOOSTER_DICT[opinion[1]]

            wn_polarity = 1 if wn_polarity > 1 else -1 if wn_polarity < -1 else wn_polarity

        wn_polarity = -wn_polarity if opinion[0] else wn_polarity
        return wn_polarity


# def score_opinion(opinion):
#     def getVaderSentiment(opinion):
#         # word = opinion[2] + " " + opinion[0]
#         word = opinion[0]
#         scores = analyzer.polarity_scores(word)
#         negative = True if scores["neg"] > scores["pos"] else False
#         # sentiment = scores["neg"] if negative else scores["pos"]
#         sentiment = scores["compound"]
#
#         if sentiment != 0 and opinion[2] and vader.BOOSTER_DICT.get(opinion[2]) is not None:
#             sentiment += vader.BOOSTER_DICT[opinion[2]]
#
#         # sentiment = 1 if sentiment > 1 else sentiment
#         # sentiment = -sentiment if negative else sentiment
#
#         return sentiment
#
#     def getSentiWord(opinion):
#
#         sentiment = getSent(opinion[0])
#         if sentiment is None:
#             return 0
#         if sentiment != 0 and opinion[2] and vader.BOOSTER_DICT.get(opinion[2]) is not None:
#             sentiment += vader.BOOSTER_DICT[opinion[2]]
#         sentiment = -1 if sentiment < -1 else sentiment
#         sentiment = 1 if sentiment > 1 else sentiment
#         return sentiment
#
#     count = 0
#
#     sentiment = getVaderSentiment(opinion)
#     # if sentiment == 0:
#     #     sentiment = getSentiWord(o)
#     if opinion[1]:
#         sentiment = -sentiment if sentiment > 0 else sentiment + vader.B_INCR if sentiment < 0 else sentiment + vader.B_DECR
#         # sentiment = -sentiment if sentiment >= 0 else
#
#     return sentiment
