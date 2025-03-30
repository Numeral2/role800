import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 🔹 Load environment variables (API keys)
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 🔹 Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "bge-embeddings-index"  # Replace with your Pinecone index name

# 🔹 Initialize Pinecone Index
index = pc.Index(index_name)

# 🔹 Load the Hugging Face model for embeddings (BAAI/bge-small-en)
hf_model = SentenceTransformer("BAAI/bge-small-en")

# 🔹 Function for retrieving relevant chunks from Pinecone
def retrieve(query_text, top_k=5):
    """
    Retrieve the top_k most relevant document chunks from Pinecone for a given query.
    
    :param query_text: The input query for which we want to find relevant chunks
    :param top_k: The number of relevant chunks to retrieve
    :return: A list of retrieval results (most relevant document chunks)
    """
    # Encode the query into an embedding using the Hugging Face model
    query_embedding = hf_model.encode(query_text).tolist()  # Convert to list of floats for Pinecone
    
    # Perform similarity search in Pinecone
    results = index.query(
        vector=query_embedding,  # The query vector (embedding)
        top_k=top_k,  # Number of similar results to return
        include_metadata=True  # Optionally include metadata in the results
    )
    
    return results

# Example: Query the Pinecone index for relevant chunks
query_text = "What are the benefits of a healthy breakfast?"
retrieved_results = retrieve(query_text)

# Display the retrieved results
print("Retrieved Results:")
for res in retrieved_results['matches']:
    print(f"Text: {res['metadata']['source']} | Score: {res['score']} | Content: {res['metadata'].get('content', 'No content available')}")

