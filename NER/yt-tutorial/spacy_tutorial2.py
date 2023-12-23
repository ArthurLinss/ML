import spacy 


nlp = spacy.load("en_core_web_sm")
with open ("data/wiki_us.txt", "r") as f:
    text = f.read()
doc = nlp(text)
sentence1 = list(doc.sents)[0]
sentence2 = list(doc.sents)[1]

# similarity of objects (words, tokens, docs), syntax is similar
sim = sentence1.similarity(sentence2)
print("similarity: ", sim)



