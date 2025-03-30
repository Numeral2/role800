import os
import time
import streamlit as st
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Index
import pdfplumber
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (Pinecone API Key, OpenAI API Key, etc.)
load_dotenv()

# Initialize Pinecone with your API Key
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

# Initialize the Pinecone index
index_name = "your_index_name"  # Set your Pinecone index name
index = Index(index_name)

# Initialize the Hugging Face embedding model (all-MiniLM-L6-v2)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Using all-MiniLM-L6-v2

st.title("Chat with Your PDF (GPT-4o-mini & Pinecone)")

# User inputs for API keys
pinecone_api_key = st.text_input("Enter your Pinecone API Key:", type="password")
pinecone_index_name = st.text_input("Enter your Pinecone Index Name:")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Function to extract text from PDF using pdfplumber and chunk it
def extract_text_from_pdf(pdf_path, chunk_size=500):
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                for idx, chunk in enumerate(chunks):
                    text_chunks.append((page_num, idx, chunk))
    return text_chunks

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

    st.success("File processed and stored in Pinecone.")

# User uploads PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully. Processing...")

    # Ingest the PDF into Pinecone
    ingest_pdf(file_path)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": "You are an AI assistant."})

# Chat input
prompt = st.text_input("Ask a question about the document:")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve context from Pinecone
    query_embedding = embedding_model.encode(prompt)
    results = index.query(query_embedding, top_k=3, include_metadata=True)

    context = "\n".join([f"Page {match['metadata']['page']}, Chunk {match['metadata']['chunk']}: {match['metadata']['content']}" for match in results['matches']])

    system_prompt = f"Context: {context}\n\nAnswer the user's question based on the above context."
    st.session_state.messages.append({"role": "system", "content": system_prompt})

    # Invoke GPT-4o-mini for response
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    result = response["choices"][0]["message"]["content"]

    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})

else:
    st.warning("Please enter your API keys to proceed.")
