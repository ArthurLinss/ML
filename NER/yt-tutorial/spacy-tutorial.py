
# spacy is a natural language processing library

import spacy 
npl = spacy.load("en_core_web_sm")


# containers: contain quantity of data about a text, e.g. doc, span and token 

with open ("data/wiki_us.txt", "r") as f:
    text = f.read() 

# create a doc container
doc = npl(text)

# check tokens
for token in doc[:10]:
    print(token)

# check sentences
for sent in list(doc.sents)[:5]:
    print(sent)


# token attributes
"""
.text

.head   -> governer of the token, e.g. following verb

.left_edge

.right_edge

.ent_type_ -> entity type as string

.iob_

.lemma_

.morph

.pos_

.dep_

.lang_
"""

sentence1 = list(doc.sents)[:0]
from spacy import displacy
#displacy.render(sentence1, style="dep")
#displacy.serve(sentence1, style="dep", port=5001)




# named entity recognition 

for ent in doc.ents:
    print(ent.text, ent.label_)

#displacy.render(doc, style="ent")
#displacy.serve(doc, style="ent", port=5001)