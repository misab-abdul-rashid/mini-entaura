
import os
import uuid

import streamlit as st
import chromadb

from dotenv import load_dotenv

from src.db import get_connection
from src.ingest import ingest_document
from src.rag import ask_question


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

DOCUMENTS_FOLDER = os.path.abspath("./documents")

os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="Mini EntAura",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# 2. DATABASE CONNECTION
# ==================================================

def get_db_connection():
    return get_connection()


# ==================================================
# 3. CHROMADB CONNECTION
# ==================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="business_documents",
    metadata={"hnsw:space": "cosine"}
)


# ==================================================
# 4. SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "upload_notifications" not in st.session_state:
    st.session_state.upload_notifications = []


# ==================================================
# 5. SAFE FILE CLEANUP
# ==================================================

def remove_uploaded_file(file_path):

    if not file_path:
        return False

    try:

        absolute_path = os.path.abspath(file_path)

        if os.path.commonpath(
            [DOCUMENTS_FOLDER, absolute_path]
        ) != DOCUMENTS_FOLDER:

            print("Unsafe file path. Cleanup skipped.")
            return False

        if os.path.isfile(absolute_path):
            os.remove(absolute_path)

        parent_folder = os.path.dirname(absolute_path)

        if (
            parent_folder != DOCUMENTS_FOLDER
            and os.path.commonpath(
                [DOCUMENTS_FOLDER, parent_folder]
            ) == DOCUMENTS_FOLDER
        ):

            try:
                os.rmdir(parent_folder)
            except OSError:
                pass

        return True

    except Exception as error:

        print("Physical file cleanup failed:", error)
        return False


# ==================================================
# 6. DOCUMENT MANAGEMENT
# ==================================================

def get_documents():

    connection = get_db_connection()
    cursor = None

    try:

        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                id,
                filename,
                file_type,
                total_chunks,
                status,
                uploaded_at
            FROM documents
            WHERE status != 'deleted'
            ORDER BY uploaded_at DESC
        """)

        return cursor.fetchall()

    finally:

        if cursor:
            cursor.close()

        connection.close()


def delete_document(document_id):

    connection = get_db_connection()
    cursor = None

    try:

        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                file_hash,
                filename,
                storage_path,
                status
            FROM documents
            WHERE id = %s
            FOR UPDATE
        """, (document_id,))

        document = cursor.fetchone()

        if not document:
            return False, "Document not found."

        file_hash, filename, storage_path, status = document

        if status == "deleted":
            return False, "Document is already deleted."

        # Remove vectors from ChromaDB.

        collection.delete(
            where={"file_hash": file_hash}
        )

        # Update PostgreSQL.

        cursor.execute("""
            UPDATE documents
            SET
                status = 'deleted',
                total_chunks = 0
            WHERE id = %s
        """, (document_id,))

        connection.commit()

        # Remove physical file.

        if storage_path:

            file_removed = remove_uploaded_file(storage_path)

            if file_removed:

                return True, (
                    f"{filename} deleted successfully "
                    "from PostgreSQL, ChromaDB and storage."
                )

            return True, (
                f"{filename} was deleted from the database "
                "and ChromaDB, but physical file cleanup failed."
            )

        return True, (
            f"{filename} deleted from PostgreSQL and ChromaDB. "
            "No storage path was available."
        )

    except Exception:

        connection.rollback()
        raise

    finally:

        if cursor:
            cursor.close()

        connection.close()


# ==================================================
# 7. FILE UPLOAD AND INGESTION
# ==================================================

def process_uploaded_file(uploaded_file):

    original_filename = os.path.basename(
        uploaded_file.name.replace("\\", "/")
    )

    file_extension = os.path.splitext(
        original_filename
    )[1].lower()

    if file_extension not in [".pdf", ".txt"]:

        return "error", "Only PDF and TXT files are supported."

    if not original_filename:

        return "error", "Invalid filename."

    upload_directory = None
    file_path = None

    try:

        # Create unique storage directory.

        unique_folder = uuid.uuid4().hex

        upload_directory = os.path.join(
            DOCUMENTS_FOLDER,
            unique_folder
        )

        os.makedirs(
            upload_directory,
            exist_ok=False
        )

        file_path = os.path.join(
            upload_directory,
            original_filename
        )

        # Save uploaded file.

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        # Ingest document.

        ingestion_result = ingest_document(file_path)

        # Duplicate or already-processing document.

        if ingestion_result is False:

            cleanup_success = remove_uploaded_file(file_path)

            if cleanup_success:

                return "duplicate", (
                    f"{original_filename} was not processed. "
                    "Identical content already exists or "
                    "is currently being processed. "
                    "The extra uploaded copy was removed."
                )

            return "warning", (
                f"{original_filename} was not processed because "
                "identical content already exists or is processing. "
                "However, temporary file cleanup failed."
            )

        return "success", (
            f"{original_filename} processed successfully."
        )

    except Exception as error:

        print("Document ingestion error:", error)

        if file_path:
            remove_uploaded_file(file_path)

        return "error", (
            f"Unable to process {original_filename}. "
            "Check the terminal for details."
        )


# ==================================================
# 8. CITATION DISPLAY
# ==================================================

def display_sources(sources):

    if not sources:
        return

    st.markdown("**Sources:**")

    for index, source in enumerate(sources, start=1):

        source_name = source.get(
            "source",
            "Unknown source"
        )

        page = source.get(
            "page",
            "N/A"
        )

        excerpts = source.get(
            "excerpts",
            []
        )

        with st.expander(
            f"{index}. {source_name} — Page {page}"
        ):

            st.markdown("**Supporting passages:**")

            if excerpts:

                for excerpt_index, excerpt in enumerate(
                    excerpts,
                    start=1
                ):

                    st.markdown(
                        f"**Passage {excerpt_index}**"
                    )

                    st.markdown(f"> {excerpt}")

            else:

                st.info("No supporting passage available.")


# ==================================================
# 9. APPLICATION HEADER
# ==================================================

st.title("📚 Mini EntAura")

st.caption(
    "Enterprise Business Knowledge Assistant"
)

st.markdown(
    """
    Upload business documents and ask questions.
    Answers are generated using your uploaded documents
    with source citations.
    """
)

st.divider()


# ==================================================
# 10. SIDEBAR
# ==================================================

with st.sidebar:

    st.header("Document Management")

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button(
        "Process Documents",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning("Please select at least one document.")

        else:

            notifications = []

            for uploaded_file in uploaded_files:

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    result_type, message = process_uploaded_file(
                        uploaded_file
                    )

                    notifications.append({
                        "type": result_type,
                        "message": message
                    })

            # Save notifications before rerunning.

            st.session_state.upload_notifications = notifications

            st.rerun()

    # ----------------------------------------------
    # DOCUMENT UPLOAD NOTIFICATIONS
    # ----------------------------------------------

    if st.session_state.upload_notifications:

        st.subheader("Upload Results")

        for notification in st.session_state.upload_notifications:

            result_type = notification["type"]
            message = notification["message"]

            if result_type == "success":

                st.success(message)

            elif result_type == "duplicate":

                st.info(message)

            elif result_type == "warning":

                st.warning(message)

            else:

                st.error(message)

        if st.button(
            "Clear Upload Notifications",
            use_container_width=True
        ):

            st.session_state.upload_notifications = []
            st.rerun()

    st.divider()

    # ----------------------------------------------
    # DOCUMENT DASHBOARD
    # ----------------------------------------------

    st.subheader("Document Dashboard")

    try:

        documents = get_documents()

        total_documents = len(documents)

        total_chunks = sum(
            document[3] or 0
            for document in documents
            if document[4] == "completed"
        )

        completed_documents = sum(
            1
            for document in documents
            if document[4] == "completed"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Documents",
            total_documents
        )

        col2.metric(
            "Chunks",
            total_chunks
        )

        st.caption(
            f"Successfully processed: {completed_documents}"
        )

        if documents:

            for document in documents:

                (
                    document_id,
                    filename,
                    file_type,
                    chunks,
                    status,
                    uploaded_at
                ) = document

                with st.expander(filename):

                    st.write("File type:", file_type)
                    st.write("Chunks:", chunks)
                    st.write("Status:", status)

                    st.write(
                        "Uploaded:",
                        str(uploaded_at)
                    )

                    if st.button(
                        "Delete Document",
                        key=f"delete_{document_id}",
                        use_container_width=True
                    ):

                        try:

                            success, message = delete_document(
                                document_id
                            )

                            if success:
                                st.success(message)
                            else:
                                st.error(message)

                            st.rerun()

                        except Exception as error:

                            st.error(
                                f"Deletion failed: {error}"
                            )

        else:

            st.info("No documents uploaded yet.")

    except Exception as error:

        st.error(
            f"Unable to load document dashboard: {error}"
        )

    st.divider()

    # ----------------------------------------------
    # CLEAR CHAT
    # ----------------------------------------------

    if st.button(
        "Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()


# ==================================================
# 11. CHAT INTERFACE
# ==================================================

st.subheader("Ask Your Documents")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            display_sources(
                message.get("sources", [])
            )


# ==================================================
# 12. CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask a question about your documents..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            response = ask_question(question)

            answer = response.get(
                "answer",
                "Unable to generate an answer."
            )

            sources = response.get(
                "sources",
                []
            )

        st.markdown(answer)

        display_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
