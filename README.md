# Mental Health Chatbot using NLTK

A retrieval-based chatbot built using **NLTK** and **TF-IDF** to answer mental health–related questions.  
The chatbot uses classic NLP techniques such as text preprocessing, vectorization, and cosine similarity to retrieve the most relevant response from a curated FAQ corpus.

⚠️ This chatbot is for **educational purposes only** and does **not provide medical advice**.

---

##  Project Overview

This project demonstrates how early chatbots worked before large language models (LLMs), using:
- Text preprocessing (NLTK)
- TF-IDF feature extraction
- Cosine similarity–based retrieval
- Confidence thresholds and fallback handling

The chatbot responds to user queries by finding the most similar question or statement in a mental health FAQ corpus.

---

##  Project Architecture

User Input
↓
Text Preprocessing (lowercase, tokenize, stopwords, lemmatize)
↓
TF-IDF Vectorization
↓
Cosine Similarity Matching
↓
Best Response Selection
↓
Fallback (if confidence is low)

##  Project Structure

nltk-chatbot/
│
├── data/
│ ├── corpus.txt # Mental health FAQ corpus
│ └── error_log.txt # Logged low-confidence queries
│
├── chatbot/
│ ├── preprocess.py # Text preprocessing pipeline
│ ├── vectorizer.py # TF-IDF vectorizer
│ ├── similarity.py # Cosine similarity logic
│ └── bot.py # Chatbot controller
│
├── main.py # Interactive chat loop
├── requirements.txt
└── README.md


---

##  How It Works

1. The corpus is loaded from `corpus.txt`
2. Text is preprocessed using NLTK:
   - lowercasing
   - tokenization
   - stopword removal
   - lemmatization
3. TF-IDF vectors are created for all corpus sentences
4. User input is vectorized using the same TF-IDF space
5. Cosine similarity is computed between user input and corpus
6. The most similar response is returned
7. If similarity is below a threshold, a fallback response is shown

---

##  Example Conversation

Mental Health Chatbot
You: I feel stressed and anxious
Bot: stress and anxiety are common and it may help to take short breaks and rest.

You: I feel empty
Bot: feeling empty can be very difficult and painful. you are not alone.

You: bye
Bot: Take care of yourself. You are not alone.
