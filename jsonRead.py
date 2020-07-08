import json
import miningfunc as mf

with open('incoherents.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        print('label: ' + str(p["label"]) + "  Prediction: "+str(p["pred"]))
        mf.myprint(p["rev"])
        print('')