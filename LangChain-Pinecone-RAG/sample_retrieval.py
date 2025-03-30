import os
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings  # Correct import
from langchain.schema import Document

# Load environment variables
load_dotenv()

# 🔹 Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=pinecone_api_key, environment="us-east-1")

# 🔹 Set up Pinecone index
index_name = "quickstart"

# Ensure index exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Must match embedding model
        metric="cosine"
    )

index = pinecone.Index(index_name)

# 🔹 Initialize Hugging Face Embeddings Model
embedding_model = SentenceTransformer("BAAI/bge-small-en")

# Function to properly encode text
def embed_text(text):
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

# 🔹 Initialize Pinecone VectorStore
vector_store = Pinecone(index, embed_text)

# 🔹 Get user input for retrieval
query = input("Enter your query: ")

# Convert query into vector
query_vector = embed_text(query)

# Perform search
results = index.query(query_vector, top_k=5, include_metadata=True)

# Display results
print("🔹 RESULTS:")
for match in results["matches"]:
    print(f"Score: {match['score']:.4f}, Content: {match['metadata']}")


