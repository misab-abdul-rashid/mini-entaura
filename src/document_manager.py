
from pathlib import Path
import chromadb

from src.db import get_connection


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = (PROJECT_ROOT / "documents").resolve()

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(
    path=str(PROJECT_ROOT / "chroma_db")
)

collection = chroma_client.get_collection(
    name="business_documents"
)


def delete_document(document_id):

    connection = get_connection()
    cursor = connection.cursor()

    storage_path = None
    filename = None

    try:

        # 1. Retrieve document information
        cursor.execute(
            """
            SELECT filename, file_hash, status, storage_path
            FROM documents
            WHERE id = %s
            FOR UPDATE
            """,
            (document_id,)
        )

        document = cursor.fetchone()

        if not document:
            return False, "Document not found."

        filename, file_hash, status, storage_path = document

        if status == "deleted":
            return False, "Document is already deleted."

        # 2. Delete associated vectors from ChromaDB
        collection.delete(
            where={"file_hash": file_hash}
        )

        # 3. Update PostgreSQL
        cursor.execute(
            """
            UPDATE documents
            SET status = 'deleted',
                total_chunks = 0
            WHERE id = %s
            """,
            (document_id,)
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()

    # 4. Delete physical file after successful database commit
    if storage_path:

        try:
            file_path = Path(storage_path)

            if not file_path.is_absolute():
                file_path = PROJECT_ROOT / file_path

            file_path = file_path.resolve()

            # Prevent deletion outside the documents directory
            if not file_path.is_relative_to(DOCUMENTS_DIR):
                return True, (
                    f"{filename} was removed from the database "
                    "and vector store, but its file path was outside "
                    "the documents directory. Manual cleanup required."
                )

            if file_path.is_file():
                file_path.unlink()

                # Remove its UUID directory if empty
                parent = file_path.parent

                if parent != DOCUMENTS_DIR and parent.is_dir():
                    try:
                        parent.rmdir()
                    except OSError:
                        pass

        except OSError as error:
            return True, (
                f"{filename} was removed from the database "
                f"and vector store, but physical file deletion failed: "
                f"{error}"
            )

    return True, f"{filename} deleted successfully."
