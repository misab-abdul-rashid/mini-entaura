import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Google API key not found. Check your .env file.")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Generate an embedding
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="An enterprise AI assistant that answers questions using company documents."
)

# Display results
embedding = response.embeddings[0].values

print("Gemini API connection successful!")
print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])