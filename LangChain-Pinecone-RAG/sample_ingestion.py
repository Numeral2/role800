import os
import time
import pdfplumber
import tiktoken
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "quickstart"  # Change if needed

# Check if index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Tokenizer setup for chunking
tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 2048  # Ensures each chunk is under OpenAI’s 4096-token limit

# Function to extract and chunk PDF text
def extract_and_chunk_pdf(pdf_path):
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens), MAX_TOKENS):
                chunk_tokens = tokens[i:i + MAX_TOKENS]
                chunk_text = tokenizer.decode(chunk_tokens)
                documents.append(Document(
                    page_content=chunk_text,
                    metadata={"document_id": pdf_path, "chunk_id": f"{page_num}_{i}", "page_number": page_num}
                ))
    return documents

# Ingest the PDF
pdf_path = "your_pdf.pdf"  # Change to your actual PDF path
documents = extract_and_chunk_pdf(pdf_path)

# Generate unique IDs
uuids = [doc.metadata["chunk_id"] for doc in documents]

# Store in Pinecone
vector_store.add_documents(documents=documents, ids=uuids)
print(f"Successfully added {len(documents)} chunks to Pinecone.")
