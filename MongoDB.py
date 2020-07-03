import pymongo
import utilities as utils
import nltk

# folder = 'cran'
# tokenizer = utils.Tokenizer(('en_core_web_sm', 'english'))
# c = utils.Cranfield(folder)
#
# print(c.count())

cran = pymongo.MongoClient()
print(cran.list_database_names())

c = cran['crands']['cran_tokens']

m = {'$match': {
    'pos': 'VERB'
}}
g = {'$group': {
    '_id': {
        'partofspeech': '$pos', 'lemma': '$lemma'
    }, 'count': {'$sum': 1}
}}
m1 = {'$match': {
    'count': {'$gte': 10}
}}
s = {'$sort': {
    'count': -1
}}

print(c.find_one())
print(c.find_one())
print(c.find_one())

# for record in c.aggregate([g, m1, s]):
#     print(record)



