import os
import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore

def retrieve_data(pinecone_api_key, pinecone_index_name, query):
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment="us-east-1")
    index = pinecone.Index(pinecone_index_name)

    # Initialize Hugging Face Embeddings Model
    embedding_model = SentenceTransformer("BAAI/bge-small-en")  # Free & lightweight
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    # Retrieval
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Perform the retrieval
    try:
        results = retriever.invoke(query)  # Correct method for retrieval
    except Exception as e:
        raise Exception(f"An error occurred while retrieving results: {e}")
    
    return results

# Streamlit UI
st.title("Retrieve Information from Your PDF")

# User inputs for API keys
pinecone_api_key = st.text_input("Enter your Pinecone API Key:", type="password")
pinecone_index_name = st.text_input("Enter your Pinecone Index Name:")
query = st.text_input("Enter your query:", "")

if pinecone_api_key and pinecone_index_name and query:
    try:
        results = retrieve_data(pinecone_api_key, pinecone_index_name, query)

        # Display results
        st.subheader("Results:")
        if results:
            for i, res in enumerate(results):
                st.markdown(f"**Result {i + 1}:**")
                st.markdown(f"**Content:** {res.page_content}")
                st.markdown(f"**Metadata:** {res.metadata}")
        else:
            st.warning("No relevant results found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter all required fields to proceed.")

