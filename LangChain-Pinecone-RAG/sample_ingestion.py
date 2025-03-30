import os
import time
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

# 🔹 Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=pinecone_api_key, environment="us-east-1")

# 🔹 Set up Pinecone index
index_name = "quickstart3"

# Check if index exists, and create if not
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Ensure this matches the model's embedding size
        metric="cosine"
    )
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pinecone.Index(index_name)

# 🔹 Initialize Hugging Face Embeddings Model
embedding_model = SentenceTransformer("BAAI/bge-small-en")

# Function to properly encode text (fix ndarray serialization)
def embed_text(text):
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

# 🔹 Initialize Pinecone VectorStore
vector_store = Pinecone(index, embed_text)

# 🔹 Function to chunk text
def chunk_text(text, chunk_size=512):
    """Splits text into smaller chunks"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Example: Reading a large document (replace this with real text from a PDF)
large_document_text = """
    (Large document text here, this would be the content of your 20+ pages of PDF)
    """

# Chunk the document into smaller pieces
chunks = chunk_text(large_document_text, chunk_size=512)

# Create documents from chunks with minimal metadata
documents = [
    Document(
        page_content=chunk,
        metadata={"document_id": "doc1", "chunk_id": f"chunk{i+1}", "page_number": (i // 5) + 1}
    ) for i, chunk in enumerate(chunks)
]

# Add the chunks to Pinecone
vector_store.add_documents(documents)

print(f"✅ Successfully added {len(documents)} document chunks to Pinecone.")

# ==============================
# Query the Pinecone index
# ==============================

while True:
    query = input("\n🔍 Enter your query (or type 'exit' to quit): ")
    
    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    # Convert query into vector
    query_vector = embed_text(query)

    # Perform search in Pinecone
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)

    # Display results
    print("\n🔹 RESULTS:")
    for res in results.get("matches", []):
        print(f"✅ Score: {res['score']:.4f} | Content: {res['metadata'].get('document_id', 'N/A')}")

