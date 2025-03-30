# import basics
import os
import time
from dotenv import load_dotenv
import pinecone
import pdfplumber

# import langchain
from langchain.pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load API keys from environment or user input
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
pc = pinecone.Index(index_name)

# Check whether the index exists, and create it if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Use the dimension of OpenAI's 'text-embedding-3-small'
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# PDF Upload and extraction using pdfplumber
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    # Extract text from the uploaded PDF
    with pdfplumber.open(uploaded_pdf) as pdf:
        raw_text = ""
        for page in pdf.pages:
            raw_text += page.extract_text()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Set chunk size to ensure each chunk is under 0.5MB
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False,
    )

    # Create chunks from the extracted text
    documents = text_splitter.split_documents([Document(page_content=raw_text)])

    # Generate unique IDs for the chunks
    uuids = [f"id{i}" for i in range(len(documents))]

    # Add documents to Pinecone Vector Store
    vector_store.add_documents(documents=documents, ids=uuids)
    st.success("PDF has been successfully uploaded and stored!")

# Chat functionality with the uploaded PDF
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks."))

# Display previous messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input
prompt = st.chat_input("Ask a question about the PDF")

if prompt:
    # Add the prompt to the chat history
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # Create retriever for Pinecone-based retrieval
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

    # Add the system prompt to the message history
    st.session_state.messages.append(SystemMessage(system_prompt))

    # Initialize GPT-4o-mini model
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

    # Invoke the model with the chat history
    result = llm.invoke(st.session_state.messages).content

    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))

