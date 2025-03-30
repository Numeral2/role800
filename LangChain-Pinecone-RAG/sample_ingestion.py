import os
import time
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

# Define Pinecone index name
index_name = "quickstart3"

# Create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Change if using a different embedding model
        metric="cosine"
    )
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Connect to the Pinecone index
index = pinecone.Index(index_name)

# Load the sentence transformer embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en")

# Initialize Pinecone VectorStore
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

# ==============================
# User Query Processing
# ==============================

# Get user query input
query = input("Enter your query: ")  # Allow user input

# Generate embedding and convert to list for Pinecone
query_embedding = embedding_model.encode(query).tolist()  # ✅ Fix applied

# Perform the query
results = index.query(query=query_embedding, top_k=3, include_metadata=True)

# Display results
print("RESULTS:")
for res in results["matches"]:
    print(f"* Score: {res['score']} | Content: {res['metadata'].get('document_id', 'N/A')}")


