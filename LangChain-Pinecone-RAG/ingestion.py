import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hugging Face embedding model (BAAI/bge-small-en)
from sentence_transformers import SentenceTransformer

def ingest_pdf(pinecone_api_key, pinecone_index_name, uploaded_pdf_path):
    # Initialize Pinecone with user API key
    pc = Pinecone(api_key=pinecone_api_key)

    # Initialize Pinecone index if it doesn't exist
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=pinecone_index_name,
            dimension=384,  # Dimension for BAAI/bge-small-en
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(pinecone_index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(pinecone_index_name)

    # Initialize the Hugging Face embeddings model
    embedding_model = SentenceTransformer("BAAI/bge-small-en")
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    # Load PDF document
    loader = PyPDFLoader(uploaded_pdf_path)
    raw_documents = loader.load()

    # Split the document into chunks ensuring each chunk is less than 0.5MB
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Maximum size of each chunk (adjust as needed)
        chunk_overlap=400,  # To ensure meaningful splits between chunks
        length_function=len,
    )

    # Split the document into chunks
    documents = text_splitter.split_documents(raw_documents)

    # Generate unique IDs for each chunk
    uuids = [f"id{i}" for i in range(1, len(documents) + 1)]

    # Add the chunks to the Pinecone vector store
    vector_store.add_documents(documents=documents, ids=uuids)

    print(f"Ingestion completed successfully with {len(documents)} chunks added to Pinecone.")

# Example of calling the function
ingest_pdf(
    pinecone_api_key="your_pinecone_api_key",
    pinecone_index_name="your_index_name",
    uploaded_pdf_path="path_to_your_pdf.pdf"
)

