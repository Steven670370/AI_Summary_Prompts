def route(query):
    if len(query) < 20:
        return "local"
    return "cloud"