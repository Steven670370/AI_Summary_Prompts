from AI_agent.memory import get_high_quality_logs

knowledge_base = []

def update_knowledge():
    new_data = get_high_quality_logs(min_rating=4)

    for q, r in new_data:
        knowledge_base.append((q, r))

    return len(new_data)

def simple_match(query, text):
    return any(word in text for word in query.split())

def retrieve(query):
    results = []

    for q, r in knowledge_base:
        if any(word in q for word in query.split()):
            results.append((q, r))

    return results[:3]
