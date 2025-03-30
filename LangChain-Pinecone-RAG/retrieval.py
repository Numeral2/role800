# import basics
import os
from dotenv import load_dotenv

# import pinecone
import pinecone
from pinecone import ServerlessSpec

# import langchain
from langchain.pinecone import PineconeVectorStore
from langchain.openai import OpenAIEmbeddings
from langchain.core.documents import Document

# load environment variables
load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Check if required API keys and index name are set
if not pinecone_api_key or not openai_api_key or not index_name:
    raise ValueError("Pinecone API key, OpenAI API key, or Pinecone index name is not set!")

# Initialize Pinecone client
pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
pc = pinecone.Index(index_name)

# Initialize embeddings model (using text-embedding-3-small)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

# Initialize vector store for Pinecone
vector_store = PineconeVectorStore(index=pc, embedding=embeddings)

# Create the retriever for Pinecone-based retrieval
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},
)

# Perform a query to the retriever
query = "what is retrieval augmented generation?"
results = retriever.invoke(query)

# Show results
print("RESULTS:")
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")


