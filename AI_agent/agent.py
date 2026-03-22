from memory import save_log
from rag import retrieve
from router import route

def cloud_agent(query):
    return f"[Cloud AI simulated answer]: {query}"

def local_generate(query):
    return f"[Local GPT simulated]: {query}"

def build_rag_prompt(query, docs):
    context = "\n".join([r for _, r in docs])

    return f"""
    Answer Question based on the following information:

    {context}

    Question: {query}
    """

def agent(query):
    docs = retrieve(query)

    if docs:
        prompt = build_rag_prompt(query, docs)
        answer = local_generate(prompt)
        source = "RAG"
    else:
        decision = route(query)

        if decision == "local":
            answer = local_generate(query)
            source = "LOCAL"
        else:
            answer = cloud_agent(query)
            source = "CLOUD"

    return answer, source