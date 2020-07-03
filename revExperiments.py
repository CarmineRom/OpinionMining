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


def map_score_stars(score):
    if score < - 0.5:
        return 1
    elif score < -0.1:
        return 2
    elif score < 0.1:
        return 3
    elif score < 0.6:
        return 4
    else:
        return 5


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
            sentences.append(review[start:(token.i - 1)])
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

    # PRINT OUTLIERS

    # DETERMINE REVIEW SENTIMENT
    if len(rev_dict) > 1:
        rev_sentiments = mf.score_aspects(rev_dict)

        if len(rev_sentiments) > 0:
            stars_mapped = list(map(map_score_stars, rev_sentiments))
            total_score = np.mean(stars_mapped)

            print("Stars Mapped: " + str(stars_mapped))
            print("Stars Average: " + str(float(total_score)))
            stars_score = round((float(stars)))

            total_score = math.floor(float(total_score))
            predictions.append((str(stars_score), str(total_score)))
            print("Mapping: ", (str(stars_score), str(total_score)))

    else:
        print("No aspect/opinion pairs")

    print("------------")
    print("------------")

print(predictions)

y_true = [p[0] for p in predictions]
y_pred = [p[1] for p in predictions]
cm = cfm(y_true, y_pred)

labels = ['1', '2', '3', '4', '5']
# labels = [1,2,3,4,5]
f = seaborn.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, fmt='g')

print("REVIEWS: " + str(len(revs)))
print("Predicted: " + str(len(predictions)))
print(star_dict)
print(metrics.classification_report(y_true, y_pred, labels=labels))

# print(cm)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(cm)
# plt.title('Confusion matrix of the classifier')
# fig.colorbar(cax)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# plt.xlabel('Predicted')
# plt.ylabel('Stars')
plt.show()
