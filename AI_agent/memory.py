import sqlite3

conn = sqlite3.connect("data/logs.db")

def init_db():
    conn.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        response TEXT,
        rating INTEGER
    )
    """)
    conn.commit()

def save_log(query, response, rating):
    conn.execute(
        "INSERT INTO logs (query, response, rating) VALUES (?, ?, ?)",
        (query, response, rating)
    )
    conn.commit()

def get_high_quality_logs(min_rating=4):
    cursor = conn.execute(
        "SELECT query, response FROM logs WHERE rating >= ?",
        (min_rating,)
    )
    return cursor.fetchall()