import os
import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

def retrieve_data(pinecone_api_key, openai_api_key, pinecone_index_name, query):
    # Initialize Pinecone with user API key
    pc = Pinecone(api_key=pinecone_api_key)

    # Set the Pinecone index
    index = pc.Index(pinecone_index_name)

    # Initialize embeddings model + vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Retrieval
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5},
    )
    results = retriever.invoke(query)

    return results

# Streamlit UI
st.title("Retrieve Information from Your PDF")

# User inputs for API keys
pinecone_api_key = st.text_input("Enter your Pinecone API Key:", type="password")
pinecone_index_name = st.text_input("Enter your Pinecone Index Name:")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Query input
query = st.text_input("Enter your query:", "")

if pinecone_api_key and openai_api_key and pinecone_index_name and query:
    # Call the retrieval function with user-provided parameters
    results = retrieve_data(pinecone_api_key, openai_api_key, pinecone_index_name, query)

    # Display results
    st.subheader("Results:")
    for res in results:
        st.markdown(f"* {res.page_content} [{res.metadata}]")

else:
    st.warning("Please enter all API keys and a query to proceed.")
