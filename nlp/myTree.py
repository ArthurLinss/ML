import nltk
from nltk import Tree
from nltk.parse.generate import generate
import random

# Define a simple grammar
grammar = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> Det N
    VP -> V NP
    Det -> 'the' | 'a' | 'one'
    N -> 'cat' | 'mammal' | 'dog'
    V -> 'is'
""")

# Generate a list of all possible sentences for this grammar
sentences = list(generate(grammar, n=10))

# Select a sentence
sentence = random.choice(sentences)

sentence = " ".join(sentence)

# Tokenize the sentence
tokens = nltk.word_tokenize(sentence)

# Perform part-of-speech tagging
tagged = nltk.pos_tag(tokens)

# Create a parse tree
parser = nltk.ChartParser(grammar)
parse_tree = parser.parse(tokens).__next__()

# Pretty print the parse tree
parse_tree.pretty_print()


