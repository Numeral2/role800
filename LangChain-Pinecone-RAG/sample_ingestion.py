import os
import time
import json
import numpy as np
from dotenv import load_dotenv

# Import Pinecone and LangChain components
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "sample-index"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index initialization
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Initialize components
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ["OPENAI_API_KEY"]
)

# Sample documents (edit these as needed)
documents = [
    Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
    ),
    Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
    ),
    Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
    )
]

# Generate embeddings and prepare vectors
texts = [doc.page_content for doc in documents]
embeddings_list = embeddings.embed_documents(texts)

vectors = []
for idx, (doc, emb) in enumerate(zip(documents, embeddings_list)):
    # Ensure embedding is serializable
    if isinstance(emb, np.ndarray):
        emb = emb.tolist()
    
    vectors.append({
        "id": f"id{idx+1}",
        "values": emb,
        "metadata": doc.metadata
    })

# Batch preparation with size validation
MAX_BATCH_SIZE = 0.5 * 1024 * 1024  # 0.5MB in bytes
current_size = 0
batches = []
current_batch = []

for vector in vectors:
    vector_size = len(json.dumps(vector).encode("utf-8"))
    
    if current_size + vector_size > MAX_BATCH_SIZE:
        batches.append(current_batch)
        current_batch = [vector]
        current_size = vector_size
    else:
        current_batch.append(vector)
        current_size += vector_size

if current_batch:
    batches.append(current_batch)

# Upsert with rate limiting
for batch in batches:
    # Convert to Pinecone's tuple format (id, values, metadata)
    pinecone_batch = [(v["id"], v["values"], v["metadata"]) for v in batch]
    index.upsert(vectors=pinecone_batch)
    print(f"Inserted batch of {len(pinecone_batch)} vectors")
    time.sleep(0.5)  # Avoid rate limits

print("All documents successfully indexed!")

