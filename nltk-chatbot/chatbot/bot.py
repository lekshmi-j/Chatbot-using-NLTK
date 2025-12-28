from chatbot.vectorizer import TfidfTextVectorizer
from chatbot.similarity import find_best_match


class ChatBot:
    def __init__(self, questions, answers, threshold=0.2):
        self.questions = questions
        self.answers = answers
        self.threshold = threshold

        # fit TF-IDF ONLY on questions
        self.vectorizer = TfidfTextVectorizer()
        self.corpus_matrix = self.vectorizer.fit(questions)

    def get_response(self, user_input):
        """
        Returns chatbot response based on cosine similarity
        """

        user_vector = self.vectorizer.transform(user_input)
        best_idx, best_score = find_best_match(user_vector, self.corpus_matrix)

        if best_score < self.threshold:
            with open("data/error_log.txt", "a") as f:
                f.write(f"USER: {user_input}\n")
            return "Sorry, I didn’t understand that. Can you please rephrase?"

        # ✅ RETURN ANSWER, NOT QUESTION
        return self.answers[best_idx]
