import os
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
from langchain.embeddings import SentenceTransformerEmbedding
from langchain.schema import Document

load_dotenv()

# 🔹 Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

# 🔹 Set up Pinecone index
index_name = "quickstart"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Matching Hugging Face embedding size
        metric="cosine"
    )

index = pinecone.Index(index_name)

# 🔹 Use Hugging Face Embeddings (Replaces OpenAI)
embedding_model = SentenceTransformer("BAAI/bge-small-en")

# Define a function to convert embeddings to a list (serialization fix)
def convert_to_list(embedding):
    return embedding.tolist()  # Converts ndarray to list

embedding_function = lambda text: convert_to_list(embedding_model.encode(text))

# 🔹 Initialize Pinecone VectorStore
vector_store = Pinecone(index=index, embedding_function=embedding_function)

# 🔹 Retrieve Similar Documents
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.6},
)

# Perform a retrieval query
query = "what did you have for breakfast?"
results = retriever.retrieve(query)

print("RESULTS:")
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
