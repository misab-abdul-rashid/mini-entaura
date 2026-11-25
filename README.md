# Mini EntAura — Business Knowledge Assistant

A mini enterprise-style Retrieval-Augmented Generation (RAG) application that allows users to upload business documents and ask questions about their contents using natural language.

The application retrieves relevant information from uploaded documents and uses Google's Gemini models to generate context-grounded answers with source citations.

## Features

- **Document ingestion:** Upload PDF and TXT documents.
- **Text extraction:** Extract text and page information from PDF files.
- **Text chunking:** Split documents into smaller chunks for retrieval.
- **Vector embeddings:** Generate embeddings using Google's Gemini embedding model.
- **Vector database:** Store and retrieve document embeddings using ChromaDB.
- **RAG-based question answering:** Retrieve relevant document chunks and generate contextual answers.
- **Source citations:** Display source filenames and page references.
- **Document management:** View and delete uploaded documents.
- **Duplicate detection:** Use SHA-256 hashing to identify duplicate uploads.
- **Metadata storage:** Maintain document metadata and processing status in PostgreSQL.
- **Retrieval evaluation:** Evaluate retrieval performance using manually labelled questions.
- **Edge-case testing:** Test paraphrased, unsupported, and other challenging questions.

## Architecture

```text
                 PDF / TXT Documents
                         |
                         v
                  Text Extraction
                         |
                         v
                    Text Chunking
                         |
                         v
                  Gemini Embeddings
                         |
                         v
                      ChromaDB
                    (Vector Store)


User Question
      |
      v
Gemini Query Embedding
      |
      v
ChromaDB Similarity Search
      |
      v
Relevant Document Chunks
      |
      v
Similarity Filtering / Optional MMR
      |
      v
Gemini LLM
      |
      v
Answer with Source Citations


PostgreSQL: Document metadata and processing status
Streamlit: User interface
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| LLM | Google Gemini |
| Embedding Model | Gemini Embedding |
| Text Splitting | LangChain Text Splitters |
| Vector Database | ChromaDB |
| Relational Database | PostgreSQL |
| PDF Processing | PyPDF |
| Environment Management | Python venv |

## Project Structure

```text
entaura-mini/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
└── src/
    ├── __init__.py
    ├── cleanup.py
    ├── db.py
    ├── debug_chunks.py
    ├── document_manager.py
    ├── evaluate_answers.py
    ├── evaluate_edge_cases.py
    ├── evaluate_retrieval.py
    ├── ingest.py
    ├── inspect_chroma.py
    ├── rag.py
    ├── test_gemini.py
    ├── test_vector_search.py
    └── vector_store.py
```

Runtime directories such as `documents/`, `chroma_db/`, and `venv/` are created locally and excluded from version control.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/misab-abdul-rashid/mini-entaura.git
cd mini-entaura
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

```env
GOOGLE_API_KEY=your_google_api_key

DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
```

Replace the placeholder values with your own credentials.

**Important:** Never commit your actual `.env` file or expose API keys and database passwords.

### 5. Configure PostgreSQL

Create a PostgreSQL database matching the `DB_NAME` value in your `.env` file.

Create the document metadata table:

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    file_type VARCHAR(20),
    total_chunks INTEGER DEFAULT 0,
    status VARCHAR(30) DEFAULT 'pending',
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    storage_path TEXT
);
```

### 6. Run the application

```powershell
streamlit run app.py
```

Streamlit will provide a local URL to open the application in your browser.

## How It Works

### Document Ingestion

1. The user uploads a PDF or TXT file.
2. The application calculates its SHA-256 hash to detect duplicate content.
3. Text is extracted from the document.
4. Extracted text is divided into chunks using a recursive character text splitter.
5. Gemini generates vector embeddings for the chunks.
6. Embeddings and associated metadata are stored in ChromaDB.
7. PostgreSQL stores document metadata, chunk count, storage path, and processing status.

### Question Answering

1. The user submits a question.
2. The question is converted into a vector embedding.
3. ChromaDB retrieves relevant document chunks.
4. Similarity filtering is applied, with optional Maximal Marginal Relevance (MMR).
5. Retrieved context is passed to Gemini.
6. Gemini generates an answer grounded in the retrieved context.
7. The application displays the answer with source citations.

## Evaluation and Testing

The project includes separate evaluation scripts for retrieval performance, answer quality, and edge cases.

### Retrieval Evaluation

- Manually labelled evaluation questions: 12.
- Retrieval configurations compared: Baseline similarity search and MMR.
- Tested retrieval depths: 3, 5, and 10.
- Baseline retrieval was selected as the default configuration based on the observed evaluation results.

**Selected baseline results:**

| Metric | Top 3 | Top 5 | Top 10 |
|---|---:|---:|---:|
| Precision | 77.78% | 71.11% | 55.87% |
| Recall | 62.96% | 71.91% | 80.71% |

These results are based on a small manually labelled dataset and should not be interpreted as production-level performance guarantees.

### Answer Quality Evaluation

The answer evaluation script reported:

- Answer correctness checks: 4/4 passed.
- Grounding checks: 4/4 passed.
- Citation validity checks: 6/6 passed.
- Unsupported-question refusal checks: 2/2 passed.

### Edge-Case Evaluation

The edge-case evaluation script passed 8 out of 8 test cases, covering scenarios such as:

- Paraphrased questions.
- Multi-part questions.
- Misleading assumptions.
- Unsupported questions.
- Questions involving multiple facts.
- Resume-related retrieval.

### Large PDF Testing

A 29-page university regulation PDF was successfully ingested into the system, producing 140 chunks.

The following retrieval tests returned the expected information:

- Credit requirements for a four-semester postgraduate programme.
- Minimum duration of postgraduate programmes.
- Minimum attendance required for end-semester examinations.

The results also included document source citations and page references.

**Evaluation limitation:** These tests demonstrate that the tested workflows function. They do not establish reliability across all document types, question types, or large-scale workloads.

## Current Limitations

- Answer quality depends on the relevance and quality of retrieved document chunks.
- Embedding and generation requests are subject to Gemini API quotas and latency.
- The evaluation dataset is relatively small.
- Conflicting information across documents requires additional source-priority and version-management logic.
- The application is a learning prototype, not a production-ready enterprise deployment.

## Future Improvements

- Add document versioning and source-priority management.
- Improve retrieval quality with hybrid search.
- Add user authentication and access control.
- Introduce document-level permissions.
- Improve response evaluation with a larger benchmark dataset.
- Add monitoring, logging, and performance optimization.
- Support additional document formats.