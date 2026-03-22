from agent import agent
from memory import save_log, init_db

def run_cli():
    init_db()

    print("=== AI Agent Started (type 'exit' to quit) ===")

    while True:
        query = input("\n You: ")

        if query.lower() == "exit" or query.lower() == "q":
            break

        answer, source = agent(query)

        print(f"\n [{source}] {answer}")

        rating = input("Rate (1-5 or skip): ")

        if rating.isdigit():
            save_log(query, answer, int(rating))