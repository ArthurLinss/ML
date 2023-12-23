import spacy
from spacy.tokens import Span
from spacy.training import Example

# Define a new category set
new_labels = ["ANIMAL", "PRODUCT", "EVENT", "PUSHBACK"]  # Custom labels for entities

# Sample training data with custom entities
train_data = [
    ("I bought a new laptop.", {"entities": [(11, 22, "PRODUCT")]}),
    ("The concert was fantastic.", {"entities": [(4, 11, "EVENT")]}),
    ("I saw a beautiful cat.", {"entities": [(8, 21, "ANIMAL")]}),
    ("Lufthansa six make a short pushback.", {"entities": [(21, 35, "SHORT PUSHBACK")]}),
    ("Lufthansa six make a pushback.", {"entities": [(21, 29, "PUSHBACK")]}),
    ("Lufthansa six make a long pushback.", {"entities": [(26, 34, "LONG PUSHBACK")]}),
    ("Lufthansa six make a pushback.", {"entities": [(21, 29, "PUSHBACK")]}),
    ("Lufthansa six make a pushback area six.", {"entities": [(21, 38, "PUSHBACKsix")]}),
    ("Lufthansa six make a pushback area seven.", {"entities": [(21, 40, "PUSHBACKseven")]}),
    ("Lufthansa six make a pushback area eight.", {"entities": [(21, 39, "PUSHBACKeight")]}),

]


# Initialize a blank SpaCy model
nlp = spacy.blank("en")

# Add the NER component to the pipeline
ner = nlp.add_pipe("ner")

# Add your custom labels to the NER model
for label in new_labels:
    ner.add_label(label)

# Train the NER model with custom categories
nlp.begin_training()
for itn in range(40):  # You might want to adjust the number of iterations
    losses = {}
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], losses=losses)
    print(losses)  # Print the training loss after each iteration

# Test the trained NER model
#test_text = "I bought a beautiful cat and attended the laptop launch event after visiting a concert."
#test_text = "Oceanic make a short pushback"
test_text = "Oceanic make a pushback area eight"
#test_text = "Oceanic make a short pushback"



doc = nlp(test_text)

# Print the detected entities with custom labels
for ent in doc.ents:
    print(f"Entity: {ent.text} | Label: {ent.label_}")
