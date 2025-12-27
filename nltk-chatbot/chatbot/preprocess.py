import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# initialize stopwords and lemmatizer once
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text):
    """
    Takes raw text and returns a list of clean tokens
    Steps:
    1. Lowercase
    2. Tokenize
    3. Remove stopwords & punctuation
    4. Lemmatize
    """

    # 1. lowercase
    text = text.lower()

    # 2. tokenize
    tokens = word_tokenize(text)

    clean_tokens = []

    for token in tokens:
        # remove punctuation
        if token in string.punctuation:
            continue

        # remove stopwords
        if token in STOP_WORDS:
            continue

        # lemmatize
        lemma = LEMMATIZER.lemmatize(token)

        # remove short/invalid tokens
        if len(lemma) > 2:
            clean_tokens.append(lemma)

    return clean_tokens
