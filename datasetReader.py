import pymongo

client = pymongo.MongoClient()
yelpdb = client["yelp"]
yelpdata = yelpdb["yelpData"]

generator = yelpdata.find(projection={"review_id": True, "text": True, "stars": True})
