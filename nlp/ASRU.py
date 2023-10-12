import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import Tree


sentence = """ pushback is approved face east "
         "after pushback pull forward disconnect abeam victor  one  one nine.""" #ocean four tango
tokens = nltk.word_tokenize(sentence)
print("tokens: ", tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)

