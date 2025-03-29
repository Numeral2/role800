import os
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

# 🔹 Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

# 🔹 Set up Pinecone index
index_name = "quickstart"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=768,  # Matching Hugging Face embedding size
        metric="cosine"
    )

index = pinecone.Index(index_name)

# 🔹 Use Hugging Face Embeddings (Replaces OpenAI)
embedding_model = SentenceTransformer("BAAI/bge-small-en")
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

# 🔹 Retrieve Similar Documents
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.6},
)

results = retriever.retrieve("what did you have for breakfast?")

print("RESULTS:")
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
