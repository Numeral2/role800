import streamlit as st
import os
from dotenv import load_dotenv
import pdfplumber
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.title("Chat with Your PDF")

# Input keys
openai_api_key = st.text_input("OpenAI API Key", type="password")
pinecone_api_key = st.text_input("Pinecone API Key", type="password")
pinecone_index_name = st.text_input("Pinecone Index Name")

def initialize_pinecone():
    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=pinecone_index_name,
            dimension=1536,  # text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        
    return PineconeVectorStore(index=pc.Index(pinecone_index_name), embedding=embeddings)

if openai_api_key and pinecone_api_key and pinecone_index_name:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    vector_store = initialize_pinecone()
    
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_pdf:
        with pdfplumber.open(uploaded_pdf) as pdf:
            chunks = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    chunks.append(text)
        
        documents = [HumanMessage(content=chunk) for chunk in chunks]
        vector_store.add_documents(documents)
        st.success("PDF successfully uploaded and stored in Pinecone!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content="You are an assistant for question-answering tasks.")]
    
    for message in st.session_state.messages:
        with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
            st.markdown(message.content)
    
    prompt = st.chat_input("Ask a question about the PDF")
    
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append(HumanMessage(content=prompt))
        
        retriever = vector_store.as_retriever()
        docs = retriever.invoke(prompt)
        context = "\n".join([doc.page_content for doc in docs])
        
        system_prompt = f"""
        You are an assistant for question-answering tasks.
        Use the following retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Context: {context}
        """
        
        st.session_state.messages.append(SystemMessage(content=system_prompt))
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        result = llm.invoke(st.session_state.messages).content
        
        with st.chat_message("assistant"):
            st.markdown(result)
            st.session_state.messages.append(AIMessage(content=result))
else:
    st.warning("Please input all required API keys and index name to proceed.")
