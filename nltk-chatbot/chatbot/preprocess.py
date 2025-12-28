import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet



# initialize stopwords and lemmatizer once
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def get_synonyms(word):
    """
    Returns a set of synonyms for a word using WordNet
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

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
            synonyms = get_synonyms(lemma)
            for syn in list(synonyms)[:2]:  # limit to avoid explosion
                if syn.isalpha():
                    clean_tokens.append(syn)

    return clean_tokens
