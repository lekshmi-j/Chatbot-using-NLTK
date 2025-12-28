import pandas as pd
from chatbot.bot import ChatBot


def load_qa_data(csv_path):
    """
    Load questions and answers from CSV file
    """
    df = pd.read_csv(csv_path)

    questions = df["Questions"].astype(str).tolist()
    answers = df["Answers"].astype(str).tolist()

    return questions, answers


def main():
    data_path = "data/Mental_Health_FAQ.csv"

    # load questions & answers
    questions, answers = load_qa_data(data_path)

    # initialize chatbot
    bot = ChatBot(questions, answers)

    print("🤖 Mental Health Chatbot")
    print("Type 'bye' or 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Bot: Take care of yourself. You are not alone.")
            break

        response = bot.get_response(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()
