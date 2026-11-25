import time
import re
import os
import hashlib
import chromadb

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.db import get_connection


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="business_documents",
    metadata={"hnsw:space": "cosine"}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Gemini embedding API supports a maximum of 100
# individual contents in one batch.
EMBEDDING_BATCH_SIZE = 100

EMBEDDING_MODEL = "gemini-embedding-001"


# --------------------------------------------------
# FILE HASHING
# --------------------------------------------------

def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)

    return sha256.hexdigest()


# --------------------------------------------------
# TEXT EXTRACTION
# --------------------------------------------------

def extract_text(file_path):

    if file_path.lower().endswith(".pdf"):

        reader = PdfReader(file_path)

        return [
            (page.extract_text() or "", page_number)
            for page_number, page in enumerate(
                reader.pages,
                start=1
            )
        ]

    elif file_path.lower().endswith(".txt"):

        with open(file_path, "r", encoding="utf-8") as file:

            return [(file.read(), 1)]

    return []


# --------------------------------------------------
# BATCH EMBEDDING GENERATION
# --------------------------------------------------

def generate_embeddings(texts):

    BATCH_SIZE = 50
    MAX_RETRIES = 5
    DELAY_BETWEEN_BATCHES = 31

    all_embeddings = []

    total_chunks = len(texts)

    total_batches = (
        total_chunks + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(f"Total chunks: {total_chunks}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total batches: {total_batches}")

    for batch_number, start in enumerate(
        range(0, total_chunks, BATCH_SIZE),
        start=1
    ):

        batch_texts = texts[start:start + BATCH_SIZE]

        print(
            f"\nProcessing batch {batch_number}/{total_batches}"
        )

        for attempt in range(MAX_RETRIES):

            try:

                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch_texts
                )

                if not response.embeddings:
                    raise ValueError("No embeddings returned.")

                batch_embeddings = [
                    item.values
                    for item in response.embeddings
                ]

                if len(batch_embeddings) != len(batch_texts):
                    raise ValueError("Embedding count mismatch.")

                all_embeddings.extend(batch_embeddings)

                print(
                    f"Batch {batch_number} completed."
                )

                break

            except Exception as error:

                error_message = str(error).lower()

                is_rate_limit = (
                    "429" in error_message
                    or "resource_exhausted" in error_message
                    or "quota" in error_message
                )

                if not is_rate_limit:
                    raise

                if attempt == MAX_RETRIES - 1:
                    raise

                # Read retry delay from API error when available.
                match = re.search(
                    r"retrydelay['\"]?\s*:\s*['\"]?(\d+)s",
                    str(error),
                    re.IGNORECASE
                )

                wait_time = (
                    int(match.group(1)) + 2
                    if match
                    else min(30 * (2 ** attempt), 120)
                )

                print(
                    f"Rate limit reached. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

        # Maintain spacing between batches.
        if batch_number < total_batches:

            print(
                f"Waiting {DELAY_BETWEEN_BATCHES} seconds "
                "before the next batch..."
            )

            time.sleep(DELAY_BETWEEN_BATCHES)

    if len(all_embeddings) != total_chunks:
        raise ValueError("Embedding count mismatch.")

    print("\nAll embeddings generated successfully.")

    return all_embeddings

# --------------------------------------------------
# DOCUMENT INGESTION
# --------------------------------------------------

def ingest_document(file_path):

    file_path = os.path.abspath(file_path)

    filename = os.path.basename(file_path)

    file_type = os.path.splitext(filename)[1].lower()

    if file_type not in [".pdf", ".txt"]:
        raise ValueError("Only PDF and TXT files are supported.")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    file_hash = calculate_file_hash(file_path)

    connection = get_connection()
    cursor = connection.cursor()

    document_id = None

    try:

        # ------------------------------------------
        # STEP 1: CHECK EXISTING DOCUMENT
        # ------------------------------------------

        cursor.execute(
            """
            SELECT id, status
            FROM documents
            WHERE file_hash = %s
            FOR UPDATE
            """,
            (file_hash,)
        )

        existing_document = cursor.fetchone()

        if existing_document:

            existing_id, existing_status = existing_document

            # Already successfully processed.

            if existing_status == "completed":

                print(f"Duplicate document skipped: {filename}")

                return False

            # Avoid concurrent processing.

            elif existing_status == "processing":

                print(f"Document is already processing: {filename}")

                return False

            # Retry failed or deleted documents.

            elif existing_status in ["failed", "deleted"]:

                document_id = existing_id

                print(f"Retrying document: {filename}")

                # Remove leftover vectors.

                collection.delete(
                    where={"file_hash": file_hash}
                )

                cursor.execute(
                    """
                    UPDATE documents
                    SET filename = %s,
                        file_type = %s,
                        storage_path = %s,
                        total_chunks = 0,
                        status = 'processing'
                    WHERE id = %s
                    """,
                    (
                        filename,
                        file_type,
                        file_path,
                        document_id
                    )
                )

                connection.commit()

            else:

                raise ValueError(
                    f"Unexpected document status: {existing_status}"
                )

        # ------------------------------------------
        # STEP 2: INSERT NEW DOCUMENT
        # ------------------------------------------

        else:

            cursor.execute(
                """
                INSERT INTO documents
                (
                    filename,
                    file_hash,
                    file_type,
                    storage_path,
                    status
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    filename,
                    file_hash,
                    file_type,
                    file_path,
                    "processing"
                )
            )

            document_id = cursor.fetchone()[0]

            connection.commit()

        # ------------------------------------------
        # STEP 3: EXTRACT TEXT
        # ------------------------------------------

        pages = extract_text(file_path)

        chunks = []

        for text, page_number in pages:

            if not text.strip():
                continue

            page_chunks = text_splitter.split_text(text)

            for chunk in page_chunks:

                chunks.append({
                    "text": chunk,
                    "page": page_number,
                    "source": filename
                })

        if not chunks:

            raise ValueError(
                "No extractable text found in document."
            )

        print(f"\nExtracted chunks: {len(chunks)}")

        # ------------------------------------------
        # STEP 4: GENERATE EMBEDDINGS IN BATCHES
        # ------------------------------------------

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = generate_embeddings(texts)

        # ------------------------------------------
        # STEP 5: STORE VECTORS IN CHROMADB
        # ------------------------------------------

        collection.upsert(
            ids=[
                f"{file_hash}_chunk_{i}"
                for i in range(len(chunks))
            ],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "file_hash": file_hash,
                    "document_id": document_id
                }
                for chunk in chunks
            ]
        )

        print("Vectors stored in ChromaDB.")

        # ------------------------------------------
        # STEP 6: UPDATE POSTGRESQL
        # ------------------------------------------

        cursor.execute(
            """
            UPDATE documents
            SET total_chunks = %s,
                status = 'completed',
                storage_path = %s
            WHERE id = %s
            """,
            (
                len(chunks),
                file_path,
                document_id
            )
        )

        connection.commit()

        print("\nDocument processed successfully!")
        print(f"Filename: {filename}")
        print(f"Document ID: {document_id}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Embedding batches: {(len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE}")

        return True

    except Exception as error:

        connection.rollback()

        print(f"\nIngestion failed: {error}")

        # ------------------------------------------
        # CLEAN UP PARTIALLY STORED VECTORS
        # ------------------------------------------

        try:

            collection.delete(
                where={"file_hash": file_hash}
            )

            print("Cleaned up ChromaDB vectors.")

        except Exception as cleanup_error:

            print(
                f"Warning: ChromaDB cleanup failed: {cleanup_error}"
            )

        # ------------------------------------------
        # MARK DOCUMENT AS FAILED
        # ------------------------------------------

        if document_id is not None:

            try:

                cursor.execute(
                    """
                    UPDATE documents
                    SET status = 'failed',
                        total_chunks = 0
                    WHERE id = %s
                    """,
                    (document_id,)
                )

                connection.commit()

            except Exception as status_error:

                connection.rollback()

                print(
                    f"Could not update document status: {status_error}"
                )

        raise

    finally:

        cursor.close()
        connection.close()


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    documents_folder = "documents"

    for root, directories, filenames in os.walk(documents_folder):

        for filename in filenames:

            if filename.lower().endswith((".pdf", ".txt")):

                file_path = os.path.join(root, filename)

                try:

                    ingest_document(file_path)

                except Exception as error:

                    print(
                        f"Failed to process {filename}: {error}"
                    )

    print(
        "\nTotal records in ChromaDB:",
        collection.count()
    )
