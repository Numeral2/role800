import os
import time
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (Pinecone API Key, etc.)
load_dotenv()

# Initialize Pinecone with your API Key
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

# Initialize the Pinecone index
index_name = "your_index_name"  # Set your Pinecone index name
index = pinecone.Index(index_name)

# Initialize the Hugging Face embedding model (BAAI/bge-small-en)
embedding_model = SentenceTransformer("BAAI/bge-small-en")

# Function for Ingesting a PDF and inserting embeddings into Pinecone
def ingest_pdf(uploaded_pdf_path):
    # Load the PDF document
    loader = PyPDFLoader(uploaded_pdf_path)
    raw_documents = loader.load()

    # Split the document into chunks of text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Adjust chunk size
        chunk_overlap=400,  # Adjust chunk overlap
        length_function=len,
    )
    documents = text_splitter.split_documents(raw_documents)

    # Generate embeddings for each document chunk
    for i, doc in enumerate(documents):
        vector = embedding_model.encode(doc["text"]).tolist()  # Convert NumPy array to list
        metadata = {"source": uploaded_pdf_path, "content": doc["text"]}
        # Insert into Pinecone
        index.upsert(vectors=[(f"id_{i}", vector, metadata)])

    print("Ingestion completed successfully!")

# Example of calling the ingestion function
ingest_pdf("path_to_your_pdf.pdf")
