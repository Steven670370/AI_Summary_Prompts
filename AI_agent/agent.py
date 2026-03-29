# agent.py

from AI_agent.memory import save_log, init_db, count_logs, get_high_quality_logs

from config.config import OPENAI_API_KEY
from openai import OpenAI

from Transformer.model import MiniTransformer
from Transformer.tokenizer import WordCollection
from Transformer.similarity import generate_response


client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def cloud_agent(query):
    if not OPENAI_API_KEY:
        return "[No API Key]"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content


tokenizer = WordCollection()
model = MiniTransformer(
    vocab_size=10000,
    d_model=32,
    num_heads=4,
    d_ff=64,
    seq_len=10
)


def agent(query):
    answer, source = generate_response(query, tokenizer, model)
    return answer, source
