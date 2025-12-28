from chatbot.vectorizer import TfidfTextVectorizer
from chatbot.similarity import find_best_match


class ChatBot:
    def __init__(self, corpus_sentences, threshold=0.2):
        self.corpus_sentences = corpus_sentences
        self.threshold = threshold

        # initialize and fit TF-IDF
        self.vectorizer = TfidfTextVectorizer()
        self.corpus_matrix = self.vectorizer.fit(corpus_sentences)

    def get_response(self, user_input):
        """
        Returns chatbot response based on cosine similarity
        """

        user_vector = self.vectorizer.transform(user_input)
        best_idx, best_score = find_best_match(user_vector, self.corpus_matrix)

        if best_score < self.threshold:
            return "Sorry, I didn’t understand that. Can you please rephrase?"

        return self.corpus_sentences[best_idx]
