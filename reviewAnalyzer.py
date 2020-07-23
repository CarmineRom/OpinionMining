import mining_funcs as mf
import numpy as np
from sklearn.metrics import confusion_matrix as cfm
from sklearn import metrics
import seaborn
import matplotlib.pyplot as plt
import datasetReader
import json


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


revs = []
generator = datasetReader.generator
rev_dict = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0}

while sum(rev_dict.values()) < 5000:
    el = next(generator)
    if rev_dict[el["stars"]] < 1000:
        revs.append((el["text"], el["stars"]))
        rev_dict[el["stars"]] += 1

nlp = mf.nlp

predictions = []
incoherent_revs = []
outliers_dic = {1: 0, 2: 0, 4: 0, 5: 0}

for rev, stars in revs:

    # PREPROCESSING

    # Remove bad chars
    review = nlp(mf.preprocessChars(rev.lower()))
    print("Review:")
    myprint(review.text)
    print("")

    # Split Sentences
    sentences = []
    start = 0
    for token in review:
        if token.sent_start:
            sentences.append(review[start:token.i])
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
    # stars = float(stars)

    has_outliers = False
    if len(rev_dict) > 0:

        outliers = []
        rev_polarities = []

        print("STARS: " + str(stars))
        for aspect, opinions in rev_dict.items():
            print("Aspect: " + aspect)
            for opinion in opinions:
                phrase = (("not " if opinion[0] else "") + opinion[1] + " " + opinion[2])
                polarity = mf.get_polarity(opinion)
                print("      {0} -- {1} -- {2}".format(phrase, "NEU" if -neu_threshold < polarity < neu_threshold else
                "POS" if polarity > neu_threshold else "NEG", polarity))

                if (stars < 3 and polarity > neu_threshold) or (
                        stars > 3 and polarity < - neu_threshold):
                    outliers.append((aspect, phrase))
                    has_outliers = True

                if not - neu_threshold < polarity < neu_threshold:
                    rev_polarities.append(polarity)

        # Print Outliers
        if len(outliers) > 0:
            print("REVIEW {0} OUTLIERS:".format("POSITIVE" if stars < 3 else "NEGATIVE"))
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


            rev_label = "NEG" if stars < 3 else "POS" if stars > 3 else "NEU"
            rev_predict = classify(np.mean(rev_polarities))

            print("Label: " + rev_label + "  Prediction: " + rev_predict + " with polarity: " + str(
                np.mean(rev_polarities)))
            if (rev_label == "POS" and rev_predict == "NEG") or (rev_label == "NEG" and rev_predict == "POS"):
                incoherent_revs.append({"rev": rev, "label": rev_label, "stars": stars, "pred": rev_predict})

            predictions.append((rev_label, rev_predict))

    else:
        print("No aspect/opinion pairs")
    if has_outliers:
        outliers_dic[stars] += 1
    print("----------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------")
    input()

y_true = [p[0] for p in predictions]
y_pred = [p[1] for p in predictions]
cm = cfm(y_true, y_pred)
labels = ['NEG', "NEU", "POS"]
f = seaborn.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, fmt='g')

print("Total reviews: " + str(len(revs)))
print("Predicted: " + str(len(predictions)))
print(metrics.classification_report(y_true, y_pred))

inc_revs = {"revs": incoherent_revs}
print("Incoherent reviews: " + str(len(incoherent_revs)))
print("Outliers Stats: ", outliers_dic)

with open("Incoherent_Reviews.txt", "w") as file:
    json.dump(incoherent_revs, file)
plt.show()
