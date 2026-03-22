# agent.py

from AI_agent.memory import save_log, init_db, count_logs, get_high_quality_logs
from AI_agent.rag import retrieve
from AI_agent.router import route, MIN_TRAIN_DATA

from config.config import OPENAI_API_KEY
import os
from openai import OpenAI

from Transformer.model import MiniTransformer
from Transformer.tokenizer import WordCollection
from Transformer.generate import generate
import numpy as np

# -----------------------------
# Initialize GPT client
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

def cloud_agent(query):
    if not OPENAI_API_KEY:
        return "[No API Key, fallback to local]"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

# -----------------------------
# Initialize local MiniTransformer
# -----------------------------
tokenizer = WordCollection()
model = MiniTransformer(
    vocab_size=10000,
    d_model=32,
    num_heads=4,
    d_ff=64,
    seq_len=10
)

# -----------------------------
# Local generation logic (short prompt / helper)
# -----------------------------
def has_enough_data():
    return count_logs() >= MIN_TRAIN_DATA

def local_generate(query, max_len=5):
    if not has_enough_data():
        return None  # Not enough data, do not use local model
    # MiniTransformer only generates short prompts
    prompt_tokens = generate(model, tokenizer, query, max_len=max_len)
    return prompt_tokens

# -----------------------------
# Build RAG (retrieval-augmented) prompt
# -----------------------------
def build_rag_prompt(query, docs):
    context = "\n".join([r for _, r in docs])
    return f"""
Answer the question based on the following information:

{context}

Question: {query}
"""

# -----------------------------
# Main agent function
# -----------------------------
def agent(query):

    # 1. Retrieve related documents from memory
    docs = retrieve(query)

    if docs and len(docs) > 0:
        # If relevant docs found → build RAG prompt and send to GPT
        prompt = build_rag_prompt(query, docs)
        answer = cloud_agent(prompt)
        source = "RAG+Cloud"

    else:
        # No relevant docs → decide whether to use local model or cloud
        decision = route(query)

        if decision == "local":
            local_prompt = local_generate(query)
            if local_prompt:
                # Use local-generated short prompt to guide GPT for final answer
                answer = cloud_agent(local_prompt)
                source = "LocalPrompt+Cloud"
            else:
                # Not enough local data → fallback to GPT
                answer = cloud_agent(query)
                source = "CloudFallback"
        else:
            answer = cloud_agent(query)
            source = "CLOUD"

    return answer, source