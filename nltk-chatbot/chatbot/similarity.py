import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_best_match(user_vector, corpus_matrix):
    """
    Computes cosine similarity between user vector and all corpus vectors
    Returns:
        best_index: index of most similar corpus sentence
        best_score: similarity score
    """

    # cosine similarity: (1, N)
    similarity_scores = cosine_similarity(user_vector, corpus_matrix)

    # flatten array
    similarity_scores = similarity_scores.flatten()

    # index of highest similarity
    best_index = np.argmax(similarity_scores)
    best_score = similarity_scores[best_index]

    return best_index, best_score
