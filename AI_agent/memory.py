import sqlite3

DB_PATH = "data/logs.db"
conn = sqlite3.connect(DB_PATH)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            response TEXT,
            rating INTEGER,
            used BOOLEAN DEFAULT 0
        )
        """)

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

def count_logs(min_rating=0):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM logs WHERE rating >= ?",
            (min_rating,)
        )
        return cursor.fetchone()[0]