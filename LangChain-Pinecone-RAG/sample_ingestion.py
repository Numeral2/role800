import os
import time
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
from langchain.schema import Document

load_dotenv()

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

# Initialize Pinecone database
index_name = "quickstart3"  # Change if desired

# Check if index exists, and create if not
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Change based on the model used (768 for small models, 1024+ for larger)
        metric="cosine"
    )
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pinecone.Index(index_name)

# Initialize Hugging Face embeddings model
embedding_model = SentenceTransformer("BAAI/bge-small-en")  # Lightweight & free

# Function to convert embeddings to list (serialization fix)
def convert_to_list(embedding):
    return embedding.tolist()  # Converts ndarray to list

# Function to chunk documents
def chunk_text(text, chunk_size=512):
    """Splits text into smaller chunks"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Example: Reading a large document (you can replace this with PDF text extraction)
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

# Generate unique ids for each chunk
uuids = [doc.metadata["chunk_id"] for doc in documents]

# Convert the document chunks to embeddings (convert to list for serialization)
embeddings = [convert_to_list(embedding_model.encode(doc.page_content)) for doc in documents]

# Add the chunks to Pinecone with minimal metadata
index.upsert(vectors=zip(uuids, embeddings), namespace="default")

print(f"Successfully added {len(documents)} document chunks to Pinecone.")

