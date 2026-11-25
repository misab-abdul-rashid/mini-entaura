import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def get_all_documents():

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            SELECT
                id,
                filename,
                file_type,
                total_chunks,
                status,
                uploaded_at
            FROM documents
            ORDER BY uploaded_at DESC
        """)

        return cursor.fetchall()

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":

    try:
        connection = get_connection()

        print("PostgreSQL connected successfully!")

        cursor = connection.cursor()
        cursor.execute("SELECT version();")

        print(cursor.fetchone()[0])

        cursor.close()
        connection.close()

    except Exception as error:
        print("Database connection failed:", error)