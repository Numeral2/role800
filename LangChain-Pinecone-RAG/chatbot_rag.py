import streamlit as st
import os
from dotenv import load_dotenv
import pinecone
import pdfplumber
from langchain.vectorstores import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.title("Chat with Your PDF")

# Input keys and index
pinecone_api_key = st.text_input("Pinecone API Key", type="password")
openai_api_key = st.text_input("OpenAI API Key", type="password")
index_name = st.text_input("Pinecone Index Name")

if pinecone_api_key and openai_api_key and index_name:
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
    index = pinecone.Index(index_name)

    # Use OpenAI's smaller embedding model (text-embedding-3-small)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding_function=embeddings)

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

        # Store chunks as embeddings in Pinecone
        for chunk in chunks:
            vector_store.add_texts([chunk])

        st.success("PDF successfully uploaded and stored!")

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

        # Create the retriever for Pinecone-based retrieval
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.5})
        docs = retriever.invoke(prompt)
        docs_text = "".join([d.page_content for d in docs])

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
    st.warning("Please input your Pinecone API key, OpenAI API key, and Pinecone Index name to proceed.")

