import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_pdf(pinecone_api_key, openai_api_key, pinecone_index_name, uploaded_pdf_path):
    # Initialize Pinecone with user API key
    pc = Pinecone(api_key=pinecone_api_key)

    # Initialize Pinecone index
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=pinecone_index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(pinecone_index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(pinecone_index_name)

    # Initialize embeddings model + vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Load PDF document from the uploaded path
    loader = PyPDFDirectoryLoader(uploaded_pdf_path)  # Path to the directory containing PDF(s)

    raw_documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False,
    )

    documents = text_splitter.split_documents(raw_documents)

    # Generate unique IDs for each document chunk
    uuids = [f"id{i}" for i in range(1, len(documents) + 1)]

    # Add the chunks to the Pinecone vector store
    vector_store.add_documents(documents=documents, ids=uuids)

    print("Ingestion completed successfully!")

