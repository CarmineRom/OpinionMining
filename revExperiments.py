import csv
import re
import miningfunc as mf
import numpy as np
from sklearn.metrics import confusion_matrix as cfm
from sklearn import metrics
import seaborn
import matplotlib.pyplot as plt
import math

file = open("rev0-5000.csv")
reader = csv.DictReader(file, fieldnames=["id", "review", "stars"])
revs = []

star_dict = {"1.0": 0, "2.0": 0, "3.0": 0, "4.0": 0, "5.0": 0}

# Skip first line
next(reader)

for row in reader:
    # if star_dict[row["stars"]] < 501:
    #     revs.append((row["review"], row["stars"]))
    #     star_dict[row["stars"]] += 1
    # if row["stars"] == "1.0":
    #     revs.append((row["review"], row["stars"]))
    revs.append((row["review"], row["stars"]))

nlp = mf.nlp

predictions = []
incoherent_revs = []

for rev, stars in revs:
    print("Review:")

    # PREPROCESS

    # Remove bad chars
    review = nlp(mf.preprocessChars(rev))
    mf.myprint(review.text)
    print("")

    # Split Sentences
    sentences = []
    start = 0
    for token in review:
        if token.sent_start:
            sentences.append(review[start:(token.i)])
            start = token.i
        if token.i == len(review) - 1:
            sentences.append(review[start:(token.i + 1)])

    # EXTRACT AO_DICT
    # print(sentences)
    rev_dict = {}
    for sentence in sentences:
        sentence_dict = mf.extract_oa_dict(sentence)
        for key, values in sentence_dict.items():
            if rev_dict.get(key) is None:
                rev_dict[key] = values
            else:
                rev_dict[key] = set(values + list(rev_dict[key]))

    # DETERMINE REVIEW POLARITY
    neu_threshold = 0.15
    stars_score = round((float(stars)))
    if len(rev_dict) > 0:

        outliers = []
        rev_polarities = []

        print("STARS: " + stars)
        for aspect, opinions in rev_dict.items():
            print("Aspect: " + aspect)
            for opinion in opinions:
                phrase = (("not " if opinion[0] else "") + opinion[1] + " " + opinion[2])
                polarity = mf.get_polarity(opinion)
                print("      {0} -- {1} -- {2}".format(phrase, "NEU" if -neu_threshold < polarity < neu_threshold else
                                                                "POS" if polarity > neu_threshold else "NEG", polarity))

                if (stars_score < 3 and polarity > neu_threshold) or (
                        stars_score > 3 and polarity < - neu_threshold):
                    outliers.append((aspect, phrase))

                if not - neu_threshold < polarity < neu_threshold:
                    rev_polarities.append(polarity)

        # Print Outliers
        if len(outliers) > 0:
            print("REVIEW {0} OUTLIERS:".format("POSITIVE" if stars_score < 3 else "NEGATIVE"))
            for aspect, opinion in outliers:
                print("Aspect: {0} -- Opinion: {1}".format(aspect, opinion))
        print(".......................................")

        # Classify Review
        if len(rev_polarities) > 0:
            def classify(score):
                if score < - neu_threshold:
                    return "NEG"
                elif -neu_threshold < score < neu_threshold:
                    return "NEU"
                else:
                    return "POS"


            rev_label = "NEG" if stars_score < 3 else "POS" if stars_score > 3 else "NEU"
            rev_predict = classify(np.mean(rev_polarities))

            print("Label: " + rev_label + "  Prediction: " + rev_predict + " with polarity: " + str(
                np.mean(rev_polarities)))
            if (rev_label == "POS" and rev_predict == "NEG") or (rev_label == "NEG" and rev_predict == "POS"):
                incoherent_revs.append({"rev": rev, "label": rev_label, "pred": rev_predict})

            predictions.append((rev_label, rev_predict))

    else:
        print("No aspect/opinion pairs")

    print("----------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------")

print(predictions)

y_true = [p[0] for p in predictions]
y_pred = [p[1] for p in predictions]
cm = cfm(y_true, y_pred)

labels = ['NEG', "NEU", "POS"]

f = seaborn.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, fmt='g')

print("REVIEWS: " + str(len(revs)))
print("Predicted: " + str(len(predictions)))
print(metrics.classification_report(y_true, y_pred))

import json

inc_revs = {}
inc_revs["revs"] = incoherent_revs
print("INCOHERENTS: " + str(len(incoherent_revs)))
print("NEU " + str(count_neu))
print("Not NEU " + str(count_notNeu))
with open("incoherents.txt", "w") as file:
    json.dump(incoherent_revs, file)
    # for rev, stars in incoherent_revs:
    #     print(stars)
    #     mf.myprint(rev)
    #     print("-----------------------------------------------")
    #     file.write(stars+" , "+rev+"\n")
plt.show()
