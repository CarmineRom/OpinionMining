import pymongo
import csv

client = pymongo.MongoClient()
yelpdb = client["yelp"]
yelpdata = yelpdb["yelpData"]
# texts = [(el["review_id"], el["text"], el["stars"]) for el in
#          yelpdata.find(projection={"text": True, "review_id": True, "stars": True})]

generator = yelpdata.find(projection={"text": True, "review_id": True, "stars": True})

# file = open("rev0-5000.csv", mode="w")
# filewriter = csv.DictWriter(file, fieldnames=["id", "review", "stars"])
# filewriter.writeheader()
# for (id, t, s) in texts[:5000]:
#     filewriter.writerow({"id": id, "review": t, "stars": s})

# AGGREGATION
# g = {
#     "$group": {
#         "_id": "$user_id",
#         "count": {"$sum": 1}
#     }
# }
# print(yelpdata.find_one())
# i = 0
# for el in yelpdata.aggregate([g], allowDiskUse=True):
#     print(el)
