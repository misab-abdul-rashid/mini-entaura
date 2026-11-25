import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="business_documents"
)

results = collection.get(
    include=["documents", "metadatas"]
)

print("Total records:", len(results["ids"]))

for i, record_id in enumerate(results["ids"]):

    print("\nRecord ID:", record_id)
    print("Metadata:", results["metadatas"][i])
    print("Document:", results["documents"][i])