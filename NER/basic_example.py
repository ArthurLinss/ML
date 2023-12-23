import spacy
from spacy import displacy
from pathlib import Path

# Load the English language model in SpaCy
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Sample text for named entity recognition
#text = "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California in 1976."
text = "Lufthansa six two one make a short pushback area six"

# Process the text using SpaCy's NLP pipeline
doc = nlp(text)

# Iterate through the entities detected in the text
for entity in doc.ents:
    print(f"Entity: {entity.text} | Type: {entity.label_}")

for word in doc:
    print(word.text,word.pos_)

# on localhost on port
#displacy.serve(doc, style = "ent", port=8000)
#displacy.serve(doc, style = "dep", port=9000)


# in jupyter:
# displacy.render(doc, style='dep', jupyter=True)


# as image
#svg = displacy.render(doc, style='ent', jupyter=False)
#file_name = 'example.svg'
#output_path = Path('./' + file_name)
#output_path.open('w', encoding='utf-8').write(svg)
