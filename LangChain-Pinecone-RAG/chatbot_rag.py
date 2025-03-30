import os
import time
import streamlit as st
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# 🔹 Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"  # Adjust if needed

# 🔹 Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# 🔹 Set up index
index_name = "quickstart3"  # Update as needed

if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=384, metric="cosine")
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pinecone.Index(index_name)

# 🔹 Load embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en")

# 🔹 Set up vector store with LangChain
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ========== 🔹 Streamlit UI ==========
st.title("🔍 Pinecone RAG Chatbot")

# Debugging: Show Pinecone connection info
with st.sidebar:
    st.write("### Debug Info")
    st.write(f"🔹 Pinecone Index: {index_name}")
    st.write(f"🔹 API Key Loaded: {'Yes' if PINECONE_API_KEY else 'No'}")
    st.write(f"🔹 Available Indexes: {pinecone.list_indexes()}")

# ========== 🔹 User Query ==========
query = st.text_input("💬 Enter your query:")

if query:
    with st.spinner("Searching..."):
        try:
            docs = retriever.get_relevant_documents(query)
            if docs:
                st.write("### 🔹 Relevant Results:")
                for doc in docs:
                    st.write(f"- {doc.page_content} (Score: {doc.metadata})")
            else:
                st.warning("No matching documents found.")
        except Exception as e:
            st.error(f"Error: {e}")

