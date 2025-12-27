from sklearn.feature_extraction.text import TfidfVectorizer
from chatbot.preprocess import preprocess_text


class TfidfTextVectorizer:
    """
    Handles TF-IDF vectorization for chatbot text
    """

    def __init__(self):
        # tokenizer expects a function that returns tokens
        self.vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
        self.tfidf_matrix = None

    def fit(self, corpus_sentences):
        """
        Fit TF-IDF on corpus sentences
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_sentences)
        return self.tfidf_matrix

    def transform(self, text):
        """
        Transform user input into TF-IDF vector
        """
        return self.vectorizer.transform([text])
