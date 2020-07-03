import miningfunc as mf
import spacy
import miningfunc as mf
from miningfunc import nlp
from miningfunc import extract_oa_dict
from miningfunc import myprint
from miningfunc import score_aspects
from spacy_symspell import SpellingCorrector
import re

phrase = " I also expected a refund for not getting a complete session \
today, due to the neglect and the fact I won't be returning for my last, she had failed to do that."

phrase2 = "One of the best Italian restaurants in a city that has several."

phrase3 = "Worst Walmart in Toronto."
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
print(mf.extract_oa_dict(nlp(p)))
spacy.displacy.serve(nlp(p), style="dep", page="true")
