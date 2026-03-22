from AI_agent.agent import agent
from AI_agent.memory import save_log, init_db
from Transformer.loss import update_knowledge
from Transformer.model import model
from Transformer.tokenizer import tokenizer
from Transformer.model import load_model_weights

def run_cli():
    init_db()
    load_model_weights(model)

    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        # generate answer
        answer, source = agent(query)

        print(f"\n[{source}] {answer}")

        rating = input("Rate (1-5 or skip): ")
        if rating.isdigit():
            save_log(query, answer, int(rating))

            learned = update_knowledge(model)
            if learned > 0:
                print(f"Learned {learned} new knowledge!")