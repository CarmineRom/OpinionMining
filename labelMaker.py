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


input_file = open('Incoherent_Reviews.txt')

labeled_dataset = []
print("For each review type:\n"
      "'T' - if there is a temporal problem\n"
      "'B' - if reviewer has been incoherent "
      "'N' - if incorrectly predicted and should not be added to dataset")
data = json.load(input_file)
for p in data[:3]:

    myprint(p["rev"])
    print()
    print('Stars_label: ' + str(p["label"]) + "  Prediction: " + str(p["pred"]))
    print()
    label = input("Label: ")
    if not label == "N":
        labeled_dataset.append({"rev": p["rev"], "label": label})

with open("Incoherent_Dataset.txt", "w") as file:
    json.dump(labeled_dataset, file)
