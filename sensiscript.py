from nltk.corpus import wordnet as wn
import csv

file = open("sentiword.txt")
newfile = open("sentiDict.txt", "w")
filewriter = csv.writer(newfile, delimiter=",")
filewriter.writerow(["POS", "ID", "pos_score", "neg_score"])
for line in file:
    l = line.split()

    # print(l)
    # print(wn.synset_from_pos_and_offset(l[0], int(l[1])))
    if len(l) > 3:
        filewriter.writerow([l[0], l[1], l[2], l[3]])
