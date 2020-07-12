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


def preprocessChars(sentence):
    rev = re.sub(r"[^a-zA-Z0-9.',:;?!]+", ' ', sentence)
    rev = re.sub(r"([;. ]+)([Bb])ut", " but", rev)
    rev = re.sub(r"([;., ]+)([Aa])nd", " and", rev)

    return rev


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
            is_shifted = False

            for adj_child in token.children:
                if adj_child.dep_ == "advmod":
                    modifier = adj_child.text
                if adj_child.dep_ == "neg":
                    is_shifted = True

            # AMOD-NOUN CASE
            if token.dep_ in ["amod", "compound"]:

                noun = token.head
                if noun.pos_ in ["NOUN", "PROPN"]:

                    # Check if head is a compound noun
                    noun_text = noun.text
                    for noun_child in noun.children:
                        if noun_child.dep_ == "neg":
                            is_shifted = True
                        if noun_child.dep_ == "compound":
                            noun_text = noun_child.text + " " + noun_text

                    if noun.dep_ in ["attr", "dobj"] and noun.head.pos_ in ["AUX", "VERB"]:
                        noun_verb = noun.head
                        for verb_child in noun_verb.children:
                            if verb_child.dep_ == "neg":
                                is_shifted = True

                    addPair(noun_text, (is_shifted, modifier, token.text))

                    # Propagate to subject conjuction
                    for noun_child in noun.children:
                        if noun_child.dep_ == "conj" and noun_child.pos_ in ["NOUN", "PROPN"]:
                            addPair(noun_child.text, (is_shifted, modifier, token.text))

                    # Propagate to adjective conjuction
                    for adj_child in token.children:
                        if adj_child.dep_ == "conj" and adj_child.tag_ in ["JJ", "JJS"]:
                            addPair(noun_text, (is_shifted, modifier, adj_child.text))

            # ACOMP CASE
            if token.dep_ in ["acomp", "pobj", "attr"]:
                verb = token.head
                while (verb.pos_ != "VERB" and verb.pos_ != "AUX") and verb.dep_ == "conj":
                    verb = verb.head

                for verb_child in verb.children:
                    if verb_child.dep_ == "neg":
                        is_shifted = True

                for verb_child in verb.children:
                    if verb_child.dep_ == "nsubj":
                        subject = verb_child

                        # Check if subject is a compound noun
                        noun_text = subject.text
                        for noun_child in subject.children:
                            if noun_child.dep_ == "compound":
                                noun_text = noun_child.text + " " + noun_text

                        addPair(noun_text, (is_shifted, modifier, token.text))

                        # Propagate to subject conjuction
                        for subj_child in subject.children:
                            if subj_child.dep_ == "conj" and subj_child.pos_ == "NOUN":
                                addPair(subj_child.text, (is_shifted, modifier, token.text))

                        # Propagate to adjective conjuction
                        for adj_child in token.children:
                            if adj_child.dep_ == "conj" and adj_child.tag_ in ["JJ", "JJS"]:
                                modifier = ""
                                is_shifted = False
                                for subchild in adj_child.children:
                                    if subchild.dep_ == "neg":
                                        is_shifted = True
                                    if subchild.dep_ == "advmod":
                                        modifier = subchild.text

                                addPair(noun_text, (is_shifted, modifier, adj_child.text))

    return oa_dict


def get_polarity(opinion):
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

    # Both Vader and SentiWord don't give correct polarity for words expensive and cheap (Very important!!)
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
        if opinion[1] and not -0.2 < wn_polarity < 0.2:
            if vader.BOOSTER_DICT.get(opinion[1]) is not None:
                wn_polarity += vader.BOOSTER_DICT[opinion[1]] if wn_polarity > 0 else -vader.BOOSTER_DICT[opinion[1]]

            wn_polarity = 1 if wn_polarity > 1 else -1 if wn_polarity < -1 else wn_polarity

        wn_polarity = -wn_polarity if opinion[0] else wn_polarity
        return wn_polarity
