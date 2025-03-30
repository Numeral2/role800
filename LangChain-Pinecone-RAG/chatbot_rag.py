import streamlit as st
import os
from dotenv import load_dotenv
import faiss
import pdfplumber
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.title("Chat with Your PDF")

# Input keys
openai_api_key = st.text_input("OpenAI API Key", type="password")

if openai_api_key:
    # Initialize the OpenAI embedding model (text-embedding-3-small)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    # Initialize FAISS index
    dimension = 1536  # Dimension of embeddings for 'text-embedding-3-small'
    index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean) distance for similarity search

    # Handle PDF upload
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_pdf:
        # Read PDF and split into chunks using pdfplumber
        with pdfplumber.open(uploaded_pdf) as pdf:
            chunks = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    chunks.append(text)

        # Generate embeddings for each chunk and add to FAISS index
        embeddings_list = []
        for chunk in chunks:
            embedding = embeddings.embed_text(chunk)
            embeddings_list.append(embedding)

        # Convert embeddings list to numpy array and add to FAISS index
        embeddings_array = np.array(embeddings_list).astype(np.float32)
        index.add(embeddings_array)

        st.success("PDF successfully uploaded and stored in FAISS!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks."))

    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Create a prompt for the user
    prompt = st.chat_input("Ask a question about the PDF")

    # Did the user submit a prompt?
    if prompt:
        # Add the user message to the history
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append(HumanMessage(prompt))

        # Generate embedding for the user's prompt
        query_embedding = embeddings.embed_text(prompt)

        # Search the FAISS index for the most similar document chunks
        query_embedding = np.array(query_embedding).astype(np.float32)
        distances, indices = index.search(query_embedding.reshape(1, -1), k=3)

        # Retrieve the relevant chunks
        docs = []
        for idx in indices[0]:
            docs.append(chunks[idx])

        docs_text = "".join(docs)

        # Create the system prompt for the assistant
        system_prompt = f"""
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Context: {docs_text}
        """

        # Add system prompt to message history
        st.session_state.messages.append(SystemMessage(system_prompt))

        # Initialize GPT-4o-mini model
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

        # Invoke the model with the message history
        result = llm.invoke(st.session_state.messages).content

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(result)
            st.session_state.messages.append(AIMessage(result))

else:
    st.warning("Please input your OpenAI API key to proceed.")

