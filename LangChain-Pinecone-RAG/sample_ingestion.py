import os
import time
import pdfplumber
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# 🔹 Load API keys from environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 🔹 Initialize Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "bge-embeddings-index"

# 🔹 Check if the index already exists, if not, create it
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # Correct dimension for BAAI/bge-small-en
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for the index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# 🔹 Load the BAAI embedding model
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

def ingest_pdf(pdf_path):
    # Split the PDF into chunks
    chunks = split_pdf_to_chunks(pdf_path)

    # Generate documents and metadata from chunks
    documents = [Document(page_content=chunk, metadata={"source": "pdf"}) for chunk in chunks]

    # Generate unique IDs for each chunk
    uuids = [f"id{i}" for i in range(1, len(documents) + 1)]

    # Convert embeddings to list of floats
    document_embeddings = [hf_model.encode(doc.page_content).tolist() for doc in documents]

    # Prepare data for upsert (ID, embeddings, metadata)
    vectors = list(zip(uuids, document_embeddings, [{"source": doc.metadata["source"]} for doc in documents]))

    # Ensure embeddings are lists of floats
    vectors = [(id, embedding, metadata) for id, embedding, metadata in vectors if isinstance(embedding, list) and all(isinstance(e, float) for e in embedding)]

    # Upsert the data into Pinecone
    index.upsert(vectors)

    print(f"{len(documents)} PDF chunks have been successfully ingested into Pinecone!")

# Example: Ingesting a PDF
pdf_path = "path_to_your_pdf_file.pdf"  # Path to your PDF file
ingest_pdf(pdf_path)

# ========== 🔹 Sample Retrieval from Pinecone ==========

def retrieve(query_text, top_k=5):
    """
    Retrieve the top_k most relevant document chunks from Pinecone for a given query.
    
    :param query_text: The input query for which we want to find relevant chunks
    :param top_k: The number of relevant chunks to retrieve
    :return: A list of retrieval results (most relevant document chunks)
    """
    # Encode the query into an embedding using the Hugging Face model
    query_embedding = hf_model.encode(query_text).tolist()  # Convert to list of floats for Pinecone
    
    # Perform similarity search in Pinecone
    results = index.query(
        vector=query_embedding,  # The query vector (embedding)
        top_k=top_k,  # Number of similar results to return
        include_metadata=True  # Optionally include metadata in the results
    )
    
    return results

# Example: Query the Pinecone index for relevant chunks
query_text = "What are the benefits of a healthy breakfast?"
retrieved_results = retrieve(query_text)

# Display the retrieved results
print("Retrieved Results:")
for res in retrieved_results['matches']:
    print(f"Text: {res['metadata']['source']} | Score: {res['score']} | Content: {res['metadata'].get('content', 'No content available')}")
