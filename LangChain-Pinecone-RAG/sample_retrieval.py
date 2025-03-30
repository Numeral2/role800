import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 🔹 Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 🔹 Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "bge-embeddings-index"

# 🔹 Connect to the Pinecone index
index = pc.Index(index_name)

# 🔹 Load BAAI embedding model
hf_model = SentenceTransformer("BAAI/bge-small-en")

# 🔹 Query function
def query_pinecone(query, top_k=3):
    query_embedding = hf_model.encode(query).tolist()  # Convert query to embedding
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Print the results
    if results.get("matches"):
        print("### Relevant Results:")
        for match in results["matches"]:
            print(f"- {match['metadata']['source']}: {match['score']:.2f} | {match['metadata']}")
    else:
        print("No matching documents found.")

# ========== 🔹 Example Query ==========
query = "What did you have for breakfast?"
query_pinecone(query)

