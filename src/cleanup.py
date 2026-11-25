import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="business_documents"
)

# Remove the old record
collection.delete(
    ids=["employee_policy.txt_page1_chunk0"]
)

print("Old record deleted successfully.")
print("Remaining records:", collection.count())