import os
import streamlit as st
import pdfplumber
import pinecone
import openai
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.title("Chat with Your PDF (GPT-4o-mini & Hugging Face)")

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

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment="us-east-1")
    index = pinecone.Index(pinecone_index_name)

    # Use Hugging Face embeddings model
    embedding_model = SentenceTransformer("BAAI/bge-small-en")  # Free & lightweight
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    st.success("Pinecone and Hugging Face embeddings initialized!")

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

        # Embed & store in Pinecone
        vector_store.add_documents(documents=[d["content"] for d in documents])
        st.success("File processed and stored in Pinecone.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(SystemMessage("You are an AI assistant."))

    # Chat input
    prompt = st.chat_input("Ask a question about the document:")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append(HumanMessage(prompt))

        # Retrieve context from Pinecone
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.invoke(prompt)
        context = "\n".join(d.page_content for d in docs)

        system_prompt = f"Context: {context}\n\nAnswer the user's question based on the above context."
        st.session_state.messages.append(SystemMessage(system_prompt))

        # Invoke GPT-4o-mini
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        result = response["choices"][0]["message"]["content"]

        with st.chat_message("assistant"):
            st.markdown(result)
            st.session_state.messages.append(AIMessage(result))

else:
    st.warning("Please enter your API keys to proceed.")

