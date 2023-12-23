import spacy 

# blank model
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
pipe_ana = nlp.analyze_pipes()
print(pipe_ana)

