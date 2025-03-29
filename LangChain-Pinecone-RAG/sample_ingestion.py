import os
import time
import streamlit as st
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

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
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

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

# Add the chunks to Pinecone with minimal metadata
vector_store.add_documents(documents=documents, ids=uuids)

# Streamlit app layout
st.title("Chat with Your Documents")

# User input for query
query = st.text_input("Enter your query:")

if query:
    # Generate the embedding for the query
    query_embedding = embedding_model.encode(query)  # This will give you a NumPy ndarray
    query_embedding = query_embedding.tolist()  # Convert ndarray to list

    # Perform the query to Pinecone
    results = index.query(query=query_embedding, top_k=3, include_metadata=True)

    # Display results
    st.write("Results:")
    for res in results['matches']:
        st.write(f"Document ID: {res['metadata']['document_id']}, Score: {res['score']}")
        st.write(f"Content: {res['metadata'].get('content', 'No content available')}")
