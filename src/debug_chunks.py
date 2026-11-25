
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="business_documents"
)

results = collection.get(
    include=["documents", "metadatas"]
)

print("\nTOTAL CHUNKS:", len(results["ids"]))

for chunk_id, metadata in zip(
    results["ids"],
    results["metadatas"]
):
    print("\nID:", chunk_id)
    print("METADATA:", metadata)
