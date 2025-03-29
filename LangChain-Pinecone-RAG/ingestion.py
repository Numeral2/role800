import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hugging Face embedding model (instead of OpenAI)
from sentence_transformers import SentenceTransformer

def ingest_pdf(pinecone_api_key, pinecone_index_name, uploaded_pdf_path):
    # Initialize Pinecone with user API key
    pc = Pinecone(api_key=pinecone_api_key)

    # Initialize Pinecone index
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=pinecone_index_name,
            dimension=384,  # Adjusted dimension for Hugging Face model
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(pinecone_index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(pinecone_index_name)

    # Initialize Hugging Face embeddings model + vector store
    embedding_model = SentenceTransformer("BAAI/bge-small-en")  # Hugging Face model
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    # Load PDF document from the uploaded path
    loader = PyPDFLoader(uploaded_pdf_path)

    raw_documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Adjust chunk size
        chunk_overlap=400,  # Adjust chunk overlap
        length_function=len,
    )

    documents = text_splitter.split_documents(raw_documents)

    # Generate unique IDs for each document chunk
    uuids = [f"id{i}" for i in range(1, len(documents) + 1)]

    # Add the chunks to the Pinecone vector store
    vector_store.add_documents(documents=documents, ids=uuids)

    print("Ingestion completed successfully!")

# Example of calling the function
ingest_pdf(
    pinecone_api_key="your_pinecone_api_key",
    pinecone_index_name="your_index_name",
    uploaded_pdf_path="path_to_your_pdf.pdf"
)

