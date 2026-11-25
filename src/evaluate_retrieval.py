import os
import chromadb

from dotenv import load_dotenv
from google import genai


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="business_documents",
    metadata={"hnsw:space": "cosine"}
)

EMBEDDING_MODEL = "gemini-embedding-001"

SIMILARITY_THRESHOLD = 0.40

TOP_K_VALUES = [3, 5, 10]

CANDIDATE_K = 10

LAMBDA_MULT = 0.7

# Document IDs in the current ChromaDB collection.
RESUME_DOCUMENT_IDS = [2, 4, 5]

POLICY_DOCUMENT_ID = 1

RINGGAI_DOCUMENT_ID = 6


# ==================================================
# 2. TEST QUESTIONS AND RELEVANCE LABELS
# ==================================================

# Labels use the format:
# document_id:chunk_index
#
# Example:
# 2:3 = Document ID 2, chunk 3
# 5:3 = Document ID 5, chunk 3
#
# Only chunks that contain information relevant
# to the question should be included.
#
# These labels are manually assigned based on
# the actual contents of the uploaded documents.

TEST_QUESTIONS = [

    {
        "id": "Q1",
        "question": "What technical skills and tools are mentioned in the resume?",
        "expected": [
            "2:1", "2:2", "2:3", "2:4",
            "4:2", "4:3", "4:4",
            "5:1", "5:2"
        ]
    },

    {
        "id": "Q2",
        "question": "What experience does the candidate have with REST APIs?",
        "expected": [
            "2:1", "2:7",
            "5:2", "5:6"
        ]
    },

    {
        "id": "Q3",
        "question": "What experience does the candidate have with data engineering and workflow orchestration?",
        "expected": [
            "2:2", "2:5", "2:6", "2:7", "2:8",
            "4:3", "4:7", "4:8",
            "5:5"
        ]
    },

    {
        "id": "Q4",
        "question": "How has the candidate worked on AI output evaluation and quality assessment?",
        "expected": [
            "2:3", "2:4", "2:6",
            "4:6",
            "5:1", "5:3", "5:4", "5:8"
        ]
    },

    {
        "id": "Q5",
        "question": "How did the candidate improve or automate manual reporting?",
        "expected": [
            "2:5", "2:6", "2:7", "2:8",
            "4:7", "4:8", "4:9",
            "5:5", "5:8"
        ]
    },

    {
        "id": "Q6",
        "question": "Which database technologies does the candidate know?",
        "expected": [
            "2:2",
            "4:3"
        ]
    },

    {
        "id": "Q7",
        "question": "How many annual paid leave days do full-time employees receive?",
        "expected": [
            "1:0"
        ]
    },

    {
        "id": "Q8",
        "question": "What benefits are provided to full-time employees?",
        "expected": [
            "1:0"
        ]
    },

    {
        "id": "Q9",
        "question": "Who is the CEO of the company?",
        "expected": []
    },

    {
        "id": "Q10",
        "question": "What is the company's annual revenue?",
        "expected": []
    },

    {
        "id": "Q11",
        "question": "Does the candidate have experience with healthcare-related projects?",
        "expected": [
            "2:8",
            "2:1"
        ]
    },

    {
        "id": "Q12",
        "question": "What is the company's office address?",
        "expected": []
    }

]


# ==================================================
# 3. DOCUMENT AND CHUNK IDENTIFICATION
# ==================================================

def get_chunk_index(chunk_id):

    """
    Extract numerical chunk index from ChromaDB IDs.

    Supported formats:

    filehash_chunk_0
    filehash_page1_chunk0
    """

    try:

        if "_page" in chunk_id and "_chunk" in chunk_id:

            return int(
                chunk_id.rsplit("_chunk", 1)[1]
            )

        if "_chunk_" in chunk_id:

            return int(
                chunk_id.rsplit("_chunk_", 1)[1]
            )

    except ValueError:

        return None

    return None


def get_chunk_label(chunk_id, metadata):

    """
    Generate unique evaluation label.

    Format:
    document_id:chunk_index
    """

    document_id = metadata.get("document_id")

    chunk_index = get_chunk_index(chunk_id)

    if document_id is None or chunk_index is None:
        return None

    return f"{document_id}:{chunk_index}"


# ==================================================
# 4. DISPLAY DOCUMENT INVENTORY
# ==================================================

def display_document_inventory():

    print("\n========== DOCUMENT INVENTORY ==========")

    results = collection.get(
        include=["metadatas"]
    )

    documents = {}

    for chunk_id, metadata in zip(
        results["ids"],
        results["metadatas"]
    ):

        document_id = metadata.get("document_id")

        source = metadata.get(
            "source",
            "Unknown"
        )

        file_hash = metadata.get(
            "file_hash",
            "Unknown"
        )

        key = (document_id, file_hash)

        if key not in documents:

            documents[key] = {
                "document_id": document_id,
                "filename": source,
                "file_hash": file_hash,
                "chunks": 0
            }

        documents[key]["chunks"] += 1

    available_ids = set()

    for document in documents.values():

        available_ids.add(document["document_id"])

        print(
            f"ID: {document['document_id']} | "
            f"Filename: {document['filename']} | "
            f"Chunks: {document['chunks']} | "
            f"Hash: {document['file_hash'][:12]}..."
        )

    print("\nResume document IDs:", RESUME_DOCUMENT_IDS)

    print("Policy document ID:", POLICY_DOCUMENT_ID)

    missing_resumes = [
        doc_id
        for doc_id in RESUME_DOCUMENT_IDS
        if doc_id not in available_ids
    ]

    if missing_resumes:

        raise ValueError(
            f"Missing resume document IDs: {missing_resumes}"
        )

    if POLICY_DOCUMENT_ID not in available_ids:

        raise ValueError(
            f"Policy document ID {POLICY_DOCUMENT_ID} not found."
        )


# ==================================================
# 5. COSINE SIMILARITY
# ==================================================

def cosine_similarity(vector_a, vector_b):

    if len(vector_a) != len(vector_b):
        return 0.0

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = sum(
        a * a
        for a in vector_a
    ) ** 0.5

    norm_b = sum(
        b * b
        for b in vector_b
    ) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ==================================================
# 6. RETRIEVE CANDIDATE CHUNKS
# ==================================================

def retrieve_candidates(query_embedding, candidate_k):

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(candidate_k, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances",
            "embeddings"
        ]
    )

    candidates = []

    for chunk_id, document, metadata, distance, embedding in zip(

        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["embeddings"][0]

    ):

        # ChromaDB cosine distance:
        # Lower distance means greater similarity.

        if distance > SIMILARITY_THRESHOLD:
            continue

        candidates.append({

            "id": chunk_id,

            "document": document,

            "metadata": metadata,

            "distance": distance,

            "embedding": embedding

        })

    return candidates


# ==================================================
# 7. BASELINE RETRIEVAL
# ==================================================

def baseline_retrieval(query_embedding, top_k):

    """
    Baseline:
    Retrieve the same candidate pool as MMR,
    then select the top K by similarity.

    This ensures a fair comparison.
    """

    candidates = retrieve_candidates(
        query_embedding,
        CANDIDATE_K
    )

    return candidates[:top_k]


# ==================================================
# 8. MMR RETRIEVAL
# ==================================================

def mmr_retrieval(query_embedding, top_k):

    candidates = retrieve_candidates(
        query_embedding,
        CANDIDATE_K
    )

    if not candidates:
        return []

    selected = []

    remaining = candidates.copy()

    while remaining and len(selected) < top_k:

        best_candidate = None

        best_score = float("-inf")

        for candidate in remaining:

            relevance = cosine_similarity(
                query_embedding,
                candidate["embedding"]
            )

            if selected:

                diversity_penalty = max(

                    cosine_similarity(
                        candidate["embedding"],
                        chosen["embedding"]
                    )

                    for chosen in selected

                )

            else:

                diversity_penalty = 0.0

            mmr_score = (

                LAMBDA_MULT * relevance

                -

                (1 - LAMBDA_MULT) * diversity_penalty

            )

            if mmr_score > best_score:

                best_score = mmr_score

                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)

        remaining = [

            candidate

            for candidate in remaining

            if candidate["id"] != best_candidate["id"]

        ]

    return selected


# ==================================================
# 9. EVALUATION METRICS
# ==================================================

def calculate_metrics(retrieved, expected):

    retrieved_labels = [

        get_chunk_label(
            item["id"],
            item["metadata"]
        )

        for item in retrieved

    ]

    expected_labels = set(expected)

    relevant = [

        label

        for label in retrieved_labels

        if label is not None and label in expected_labels

    ]

    # Unanswerable question.

    if not expected_labels:

        return {

            "precision": None,

            "recall": None,

            "hit": None,

            "false_positive": len(retrieved) > 0,

            "retrieved_labels": retrieved_labels,

            "relevant_count": 0

        }

    precision = (

        len(relevant) / len(retrieved)

        if retrieved else 0.0

    )

    recall = (

        len(set(relevant)) / len(expected_labels)

    )

    hit = len(relevant) > 0

    return {

        "precision": precision,

        "recall": recall,

        "hit": hit,

        "false_positive": False,

        "retrieved_labels": retrieved_labels,

        "relevant_count": len(set(relevant))

    }


# ==================================================
# 10. RUN ONE EVALUATION METHOD
# ==================================================

def evaluate_method(method_name, retrieval_function, top_k):

    print("\n----------", method_name, "----------")

    precision_scores = []

    recall_scores = []

    hit_scores = []

    false_positives = 0

    correct_rejections = 0

    unmapped_chunks = 0

    total_retrieved = 0

    for test in TEST_QUESTIONS:

        query_embedding = test["embedding"]

        retrieved = retrieval_function(
            query_embedding,
            top_k
        )

        metrics = calculate_metrics(
            retrieved,
            test["expected"]
        )

        labels = metrics["retrieved_labels"]

        total_retrieved += len(retrieved)

        unmapped_chunks += labels.count(None)

        print(
            f"\n{test['id']} | Retrieved: {labels}"
        )

        if not test["expected"]:

            if metrics["false_positive"]:

                false_positives += 1

                print("Result: FALSE POSITIVE")

            else:

                correct_rejections += 1

                print("Result: CORRECT REJECTION")

            continue

        precision_scores.append(
            metrics["precision"]
        )

        recall_scores.append(
            metrics["recall"]
        )

        hit_scores.append(
            metrics["hit"]
        )

        print(

            f"Relevant chunks: {metrics['relevant_count']} | "

            f"Precision: {metrics['precision'] * 100:.2f}% | "

            f"Recall: {metrics['recall'] * 100:.2f}% | "

            f"Hit: {metrics['hit']}"

        )

    avg_precision = (

        sum(precision_scores) / len(precision_scores)

        if precision_scores else 0

    )

    avg_recall = (

        sum(recall_scores) / len(recall_scores)

        if recall_scores else 0

    )

    avg_hit_rate = (

        sum(hit_scores) / len(hit_scores)

        if hit_scores else 0

    )

    print("\n========== SUMMARY ==========")

    print(
        f"Average Precision: {avg_precision * 100:.2f}%"
    )

    print(
        f"Average Recall: {avg_recall * 100:.2f}%"
    )

    print(
        f"Hit Rate: {avg_hit_rate * 100:.2f}%"
    )

    print("False positives:", false_positives)

    print("Correct rejections:", correct_rejections)

    print("Unmapped chunks:", unmapped_chunks)

    print("Total retrieved chunks:", total_retrieved)

    return {

        "method": method_name,

        "top_k": top_k,

        "precision": avg_precision,

        "recall": avg_recall,

        "hit_rate": avg_hit_rate,

        "false_positives": false_positives,

        "correct_rejections": correct_rejections,

        "unmapped_chunks": unmapped_chunks

    }


# ==================================================
# 11. MAIN EVALUATION
# ==================================================

def run_evaluation():

    total_chunks = collection.count()

    if total_chunks == 0:

        print("No chunks found in ChromaDB.")

        return

    print("\n======================================")

    print(" MINI ENTAURA RETRIEVAL EVALUATION")

    print("======================================")

    print("Total chunks:", total_chunks)

    print("Similarity threshold:", SIMILARITY_THRESHOLD)

    print("MMR lambda:", LAMBDA_MULT)

    print("Candidate pool:", CANDIDATE_K)

    # ----------------------------------------------
    # DISPLAY DOCUMENT INVENTORY
    # ----------------------------------------------

    display_document_inventory()

    # ----------------------------------------------
    # GENERATE QUESTION EMBEDDINGS
    # ----------------------------------------------

    print("\nGenerating question embeddings...")

    question_embeddings = {}

    for test in TEST_QUESTIONS:

        response = client.models.embed_content(

            model=EMBEDDING_MODEL,

            contents=test["question"]

        )

        if not response.embeddings:

            raise ValueError(
                f"Embedding failed for {test['id']}"
            )

        question_embeddings[test["id"]] = (

            response.embeddings[0].values

        )

    print("Question embeddings generated.")

    # ----------------------------------------------
    # ATTACH EMBEDDINGS TO QUESTIONS
    # ----------------------------------------------

    for test in TEST_QUESTIONS:

        test["embedding"] = question_embeddings[
            test["id"]
        ]

    all_results = []

    # ----------------------------------------------
    # RUN EXPERIMENTS
    # ----------------------------------------------

    methods = {

        "BASELINE": baseline_retrieval,

        "MMR": mmr_retrieval

    }

    for top_k in TOP_K_VALUES:

        print("\n\n######################################")

        print("TOP_K =", top_k)

        print("######################################")

        for method_name, retrieval_function in methods.items():

            result = evaluate_method(

                method_name,

                retrieval_function,

                top_k

            )

            all_results.append(result)

    # ----------------------------------------------
    # FINAL COMPARISON
    # ----------------------------------------------

    print("\n\n======================================")

    print(" FINAL RETRIEVAL COMPARISON")

    print("======================================")

    print(

        f"{'TOP_K':<10}"

        f"{'METHOD':<12}"

        f"{'PRECISION':<14}"

        f"{'RECALL':<14}"

        f"{'HIT RATE':<14}"

        f"{'FALSE POS':<12}"

    )

    for result in all_results:

        print(

            f"{result['top_k']:<10}"

            f"{result['method']:<12}"

            f"{result['precision'] * 100:<14.2f}"

            f"{result['recall'] * 100:<14.2f}"

            f"{result['hit_rate'] * 100:<14.2f}"

            f"{result['false_positives']:<12}"

        )

    print("\nEvaluation completed.")

    print(
        "Note: Results depend on manually assigned relevance labels."
    )


# ==================================================
# 12. EXECUTION
# ==================================================

if __name__ == "__main__":

    run_evaluation()