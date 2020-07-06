import miningfunc as mf
import spacy
import miningfunc as mf
from miningfunc import nlp
from miningfunc import extract_oa_dict
from miningfunc import myprint
from miningfunc import score_opinion
from spacy_symspell import SpellingCorrector
import re

phrase = "You can't really find anything wrong with this place, the pastas and pizzas are both amazing and high quality, the price is very reasonable, the \
owner and the staff are very friendly, if you're in downtown check this place out, a lot of people think just because it's downtown there \
are lots of options around but that's not always the case as there is also a lot of poor quality food in downtown as well."

phrase2 = " The portions were small and there wasn't much variety in broth or topping choices."

phrase3 = "I got the Paneer cheese BBQ bowl with brown rice, regular naan and tikka masala, which was very fresh and not very spicy at all I loved the green mint yogurt as well."
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

p = phrase2
print(mf.extract_oa_dict(nlp(p)))
spacy.displacy.serve(nlp(p), style="dep", page="true")
