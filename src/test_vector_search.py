import os
import chromadb
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="business_documents",
    metadata={"hnsw:space": "cosine"}
)

# Sample business documents
documents = [
    "Employees are entitled to 20 days of annual paid leave.",
    "The company provides health insurance to all full-time employees.",
    "Salary is credited on the last working day of every month."
]

# Generate embeddings
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=documents
)

embeddings = [item.values for item in response.embeddings]

# Store documents and embeddings
collection.upsert(
    ids=["doc_1", "doc_2", "doc_3"],
    documents=documents,
    embeddings=embeddings,
    metadatas=[
        {"category": "leave_policy"},
        {"category": "insurance_policy"},
        {"category": "salary_policy"}
    ]
)

print("Documents stored successfully!")

# Search using a question
question = "How many paid holidays can employees take?"

query_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

query_embedding = query_response.embeddings[0].values

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print("\nSearch question:", question)
print("\nRetrieved documents:")

for document in results["documents"][0]:
    print("-", document)