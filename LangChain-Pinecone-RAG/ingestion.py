import os
import time
import streamlit as st
import openai
from dotenv import load_dotenv
from pinecone import Index, init
import pdfplumber
import numpy as np

# Load environment variables (Pinecone API Key, OpenAI API Key, etc.)
load_dotenv()

# Streamlit interface setup
st.title("Chat with Your PDF (GPT-4o-mini & Pinecone)")

# User inputs for API keys and index name
pinecone_api_key = st.text_input("Enter your Pinecone API Key:", type="password")
pinecone_index_name = st.text_input("Enter your Pinecone Index Name:")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Set OpenAI API key
openai.api_key = openai_api_key

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

# Function for generating embeddings using OpenAI's `text-embedding-3-small` model
def generate_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=text
    )
    return response['data'][0]['embedding']

# Function for ingesting a PDF and inserting embeddings into Pinecone
def ingest_pdf(uploaded_pdf_path):
    # Extract text from PDF
    text_chunks = extract_text_from_pdf(uploaded_pdf_path, chunk_size=500)

    # Connect to Pinecone and initialize the index
    init(api_key=pinecone_api_key, environment="us-east-1")
    index = Index(pinecone_index_name)

    # Generate embeddings and insert into Pinecone
    for i, (page_num, chunk_idx, chunk) in enumerate(text_chunks):
        vector = generate_embedding(chunk)  # Get embedding using OpenAI
        metadata = {"page": page_num, "chunk": chunk_idx, "content": chunk}
        
        # Check the vector size is under 0.5 MB (500 KB max for each chunk)
        if len(np.array(vector).tobytes()) <= 400000:
            index.upsert(vectors=[(f"id_{i}", vector, metadata)])

    st.success("PDF processed and stored in Pinecone.")

# Handle PDF file upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully. Processing...")

    # Ingest the PDF into Pinecone
    ingest_pdf(file_path)

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": "You are an AI assistant."})

# Handle user input for asking questions about the document
prompt = st.text_input("Ask a question about the document:")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve context from Pinecone based on the user's query
    query_embedding = generate_embedding(prompt)  # Get embedding using OpenAI
    init(api_key=pinecone_api_key, environment="us-east-1")
    index = Index(pinecone_index_name)
    
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
