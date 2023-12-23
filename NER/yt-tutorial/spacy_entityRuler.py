import spacy 

nlp = spacy.load("en_core_web_sm")

text = "The village of Treblinka is in Poland. Treblinka was also an extermination camp."

ruler = nlp.add_pipe("entity_ruler",  before="ner") # or after?

print(nlp.analyze_pipes())

patterns = [
                {"label": "LOC", "pattern": "Treblinka"}
            ]

ruler.add_patterns(patterns)

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)