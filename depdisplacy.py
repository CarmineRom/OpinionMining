import miningfunc as mf
import spacy
import miningfunc as mf
from miningfunc import nlp
from miningfunc import extract_oa_dict
from miningfunc import myprint
from spacy_symspell import SpellingCorrector
import re

phrase = "I went to newport motors yesterday to buy a new car.  I bought 2 cars from them previously. They have a new staff now. \
The new manager there was awful no people skills what so ever.  He didn't speak when we 1st came in he just sent a \
new guy out to us who didn't know what he was doing. He had to go back and forth to the manager the whole time. \
Long story short. The manager didn't want to work with my daughter and I. He never came out of his corner. The old staff was \
great. No matter if they could help you or not they treated everyone with the best customer service. If you want to get a car \
from newport motors go to the new one on west Sahara. Avoid ( East Sahara ) like the plaque! !!!"

phrase2 = " The portions were small and there wasn't much variety in broth or topping choices."

phrase3 = "a word of advice, parking is just horrendous in the plaza which this place is situated within."
# print(phrase)
# corrector = SpellingCorrector()
# nlp.add_pipe(corrector)

#
# doc = nlp(phrase)
# for s in doc._.suggestions:
#     print(s)
# print(mf.resolveCoreference(phrase))
#
# oa_dict = extractDict(phrase)
# print(oa_dict)
# print(scoreAspects(oa_dict))

p = phrase3
print(mf.extract_oa_dict(nlp(mf.preprocessChars(p))))
spacy.displacy.serve(nlp(p), style="dep", page="true")
