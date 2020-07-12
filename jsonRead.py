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


with open('Incoherent_Reviews.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        print('label: ' + str(p["label"]) + "  Prediction: " + str(p["pred"]))
        myprint(p["rev"])
        print('')
