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
      "'i' - to add the review\n"
      "Any Key - to discard\n")

data = json.load(input_file)
for p in data:

    myprint(p["rev"])
    print()
    print("Stars:"+str(p["stars"])+"  Stars_label: " + str(p["label"]) + "  Prediction: " + str(p["pred"]))
    print()
    label = input("Label: ")
    if label == "i":
        labeled_dataset.append({"rev": p["rev"], "label": label})

with open("Incoherent_Dataset.txt", "w") as file:
    json.dump(labeled_dataset, file)
