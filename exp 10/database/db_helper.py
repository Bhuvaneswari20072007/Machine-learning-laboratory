import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, 'database')
DB_PATH = os.path.join(DB_DIR, 'predictions.db')


def get_db_connection():
    """
    Creates and returns a connection to the SQLite database.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initializes the database and creates the predictions table if it does not exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_hours REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_score REAL NOT NULL,
            sleep_hours REAL NOT NULL,
            predicted_score REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def insert_prediction(study_hours: float, attendance: float, previous_score: float, 
                      sleep_hours: float, predicted_score: float) -> int:
    """
    Inserts a newly calculated student prediction into the SQLite database.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO predictions (study_hours, attendance, previous_score, sleep_hours, predicted_score, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (study_hours, attendance, previous_score, sleep_hours, predicted_score, timestamp))
    
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def get_all_predictions(limit: int = 100) -> list[dict]:
    """
    Retrieves prediction records ordered by latest first.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, study_hours, attendance, previous_score, sleep_hours, predicted_score, timestamp
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    predictions = [
        {
            'id': row['id'],
            'study_hours': round(row['study_hours'], 2),
            'attendance': round(row['attendance'], 2),
            'previous_score': round(row['previous_score'], 2),
            'sleep_hours': round(row['sleep_hours'], 2),
            'predicted_score': round(row['predicted_score'], 2),
            'timestamp': row['timestamp']
        }
        for row in rows
    ]
    conn.close()
    return predictions


def clear_history():
    """
    Clears all prediction history.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db()
    sample_id = insert_prediction(5.0, 85.0, 70.0, 7.0, 78.52)
    print(f"Sample prediction inserted with ID: {sample_id}")
    history = get_all_predictions()
    print(f"Total history count: {len(history)}")
    print(history)
