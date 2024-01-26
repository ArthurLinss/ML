import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models

# Sample abstracts (replace with your actual abstracts)
abstracts = [
    "This is the first paper abstract.",
    "The second paper discusses important topics.",
    "In the third paper, we present new findings.",
    "This is the third paper abstract.",
    "This is the first paper abstract.",
]


# Tokenize abstracts
tokenized_abstracts = [abstract.split() for abstract in abstracts]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(abstracts)

# Calculate cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Extracting pairs with high similarity
similar_pairs = np.where(cosine_similarities > 0.5)  # Adjust the threshold as needed

print("\nSimilar Pairs:")
for i, j in zip(similar_pairs[0], similar_pairs[1]):
    if i < j:
        print(f"Pair ({i+1}, {j+1}) - Similarity: {cosine_similarities[i, j]}")

# LDA modeling
dictionary = corpora.Dictionary(tokenized_abstracts)
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_abstracts]

lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Get common words and topics for similar abstract pairs
for i, j in zip(similar_pairs[0], similar_pairs[1]):
    if i < j:
        print(f"\nCommon Words and Topics for Pair ({i+1}, {j+1}):")
        tokens_i = tokenized_abstracts[i]
        tokens_j = tokenized_abstracts[j]

        # Common words
        common_words = set(tokens_i) & set(tokens_j)
        print(f"Common Words: {common_words}")

        # Topics for each abstract
        topics_i = lda_model[dictionary.doc2bow(tokens_i)]
        topics_j = lda_model[dictionary.doc2bow(tokens_j)]

        # Common topics
        common_topics = set([topic[0] for topic in topics_i]) & set([topic[0] for topic in topics_j])
        print(f"Common Topics: {common_topics}")
