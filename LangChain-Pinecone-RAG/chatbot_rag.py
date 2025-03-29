import streamlit as st
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from PyPDF2 import PdfReader

st.title("Chat with Your PDF")

# User inputs for API keys
pinecone_api_key = st.text_input("Enter your Pinecone API Key:", type="password")
pinecone_index_name = st.text_input("Enter your Pinecone Index Name:")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if pinecone_api_key and openai_api_key and pinecone_index_name:
    st.session_state["PINECONE_API_KEY"] = pinecone_api_key
    st.session_state["PINECONE_INDEX_NAME"] = pinecone_index_name
    st.session_state["OPENAI_API_KEY"] = openai_api_key

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    st.success("Pinecone and OpenAI initialized successfully!")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully. Processing...")

        # Read PDF and extract text
        reader = PdfReader(file_path)
        pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        # Embed text into Pinecone
        vector_store.add_texts([pdf_text])
        st.success("File processed and added to the database.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks."))

    # Chat input
    prompt = st.chat_input("Ask a question about the document:")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append(HumanMessage(prompt))

        # Retrieve context
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.5})
        docs = retriever.invoke(prompt)
        docs_text = "".join(d.page_content for d in docs)

        system_prompt = f"""You are an assistant for question-answering tasks. Use the following retrieved context to answer the question. If you don't know the answer, say so. Context: {docs_text}"""
        st.session_state.messages.append(SystemMessage(system_prompt))

        # Invoke LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=1, api_key=openai_api_key)
        result = llm.invoke(st.session_state.messages).content

        with st.chat_message("assistant"):
            st.markdown(result)
            st.session_state.messages.append(AIMessage(result))

else:
    st.warning("Please enter your API keys to proceed.")
