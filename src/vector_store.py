import chromadb

# Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection
collection = client.get_or_create_collection(
    name="business_documents",
    metadata={"hnsw:space": "cosine"}
)

print("ChromaDB connected successfully!")
print("Collection:", collection.name)
print("Existing documents:", collection.count())