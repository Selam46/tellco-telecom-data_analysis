import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    """
    Create a database connection to PostgreSQL
    """
    load_dotenv()
    
    try:
        connection = psycopg2.connect(
            database="telecom_data",
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def execute_query(query, params=None):
    """
    Execute a query and return results
    """
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if query.strip().upper().startswith('SELECT'):
                return cur.fetchall()
            conn.commit()
            return True
    except Exception as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        conn.close() 