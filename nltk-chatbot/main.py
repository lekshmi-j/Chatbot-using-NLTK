from chatbot.bot import ChatBot


def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    corpus_path = "data/corpus.txt"
    corpus_sentences = load_corpus(corpus_path)

    bot = ChatBot(corpus_sentences)

    print("🤖 Mental Health Chatbot")
    print("Type 'bye' or 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip().lower()

        if user_input in ["bye", "exit", "quit"]:
            print("Bot: Take care of yourself. You are not alone.")
            break

        response = bot.get_response(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()
