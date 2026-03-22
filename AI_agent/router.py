from config.config import MIN_TRAIN_DATA
from AI_agent.memory import count_logs

def has_enough_data():
    return count_logs() >= MIN_TRAIN_DATA

def route(query):
    if len(query) < 20 and has_enough_data():
        return "local"
    return "cloud"