import os
import streamlit as st
import pdfplumber
import openai
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Index

st.title("Chat with Your PDF (GPT-4o-mini & Pinecone)")

# User inputs for API keys
pinecone_api_key = st.text_input("Enter your Pinecone API Key:", type="password")
pinecone_index_name = st.text_input("Enter your Pinecone Index Name:")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Function to extract text from PDF and chunk it
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

if pinecone_api_key and openai_api_key and pinecone_index_name:
    st.session_state["PINECONE_API_KEY"] = pinecone_api_key
    st.session_state["PINECONE_INDEX_NAME"] = pinecone_index_name
    st.session_state["OPENAI_API_KEY"] = openai_api_key

    # Initialize Pinecone client
    pinecone.init(api_key=pinecone_api_key, environment="us-east1-gcp")
    
    # Create an index if it doesn't exist
    if pinecone_index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=pinecone_index_name,
            dimension=512,  # Dimension of the embedding model (use appropriate dimension)
            metric="cosine"
        )

    # Access the index
    index = Index(pinecone_index_name)

    # Use a SentenceTransformer model for embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Can use other lightweight models

    st.success("Pinecone and embeddings model initialized!")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully. Processing...")

        # Extract and chunk text
        pdf_chunks = extract_text_from_pdf(file_path)
        documents = [
            {"content": chunk, "metadata": {"page": page_num, "chunk": idx}}
            for page_num, idx, chunk in pdf_chunks
        ]

        # Embed text and store in Pinecone
        embeddings = [embedding_model.encode(doc["content"]) for doc in documents]
        vectors_to_upsert = [
            {"id": f"{doc['metadata']['page']}_{doc['metadata']['chunk']}", "values": embedding, "metadata": doc['metadata']}
            for doc, embedding in zip(documents, embeddings)
        ]
        index.upsert(vectors=vectors_to_upsert)

        st.success("File processed and stored in Pinecone.")

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

