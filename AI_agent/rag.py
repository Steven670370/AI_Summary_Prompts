from memory import get_high_quality_logs

def simple_match(query, text):
    return any(word in text for word in query.split())

def retrieve(query):
    logs = get_high_quality_logs()

    results = []
    for q, r in logs:
        if simple_match(query, q):
            results.append((q, r))

    return results[:3]  # top 3