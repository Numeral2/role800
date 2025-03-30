import os
import time
import pdfplumber
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# 🔹 Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 🔹 Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "bge-embeddings-index"

# 🔹 Check or create index (384 dimensions for BAAI/bge-small-en embeddings)
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # Correct dimension for BAAI/bge-small-en
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# 🔹 Load BAAI embedding model
hf_model = SentenceTransformer("BAAI/bge-small-en")

# 🔹 Setup Pinecone Vector Store
vector_store = PineconeVectorStore(index=index, embedding=hf_model)

# 🔹 Function to split a PDF into chunks of less than 0.5MB (500KB)
def split_pdf_to_chunks(pdf_path, max_chunk_size_kb=500):
    chunks = []
    max_chunk_size = max_chunk_size_kb * 1024  # Convert to bytes (500KB -> 512,000 bytes)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                current_chunk = ""
                for word in text.split():
                    # Add word to current chunk
                    current_chunk += " " + word
                    # Check the current chunk's size
                    if sys.getsizeof(current_chunk.encode("utf-8")) > max_chunk_size:
                        # If chunk exceeds 0.5MB, add the current chunk and start a new one
                        chunks.append(current_chunk.strip())
                        current_chunk = word  # Start a new chunk with the current word
                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
    
    return chunks

# ========== 🔹 Document Ingestion from PDF ==========
pdf_path = "path_to_your_pdf_file.pdf"  # Path to your PDF file
chunks = split_pdf_to_chunks(pdf_path)

# Generate documents and metadata from chunks
documents = [Document(page_content=chunk, metadata={"source": "pdf"}) for chunk in chunks]

# Generate unique IDs
uuids = [f"id{i}" for i in range(1, len(documents) + 1)]

# Convert embeddings to list of floats
document_embeddings = [hf_model.encode(doc.page_content).tolist() for doc in documents]

# Insert documents into Pinecone with embeddings
vectors = list(zip(uuids, document_embeddings, [{"source": doc.metadata["source"]} for doc in documents]))

# Ensure that each embedding is a list of floats
vectors = [(id, embedding, metadata) for id, embedding, metadata in vectors if isinstance(embedding, list) and all(isinstance(e, float) for e in embedding)]

# Upsert the data into Pinecone
index.upsert(vectors)

print(f"{len(documents)} PDF chunks have been successfully ingested into Pinecone!")

