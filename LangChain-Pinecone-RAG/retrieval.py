import os
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Pinecone with the API key from environment variables
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

# Pinecone index name (replace with your actual index name)
index_name = "your_index_name"

# Initialize the Pinecone index
index = pinecone.Index(index_name)

# Initialize the Hugging Face embedding model (text-embedding-3-small)
hf_model = SentenceTransformer("text-embedding-3-small")

# Function to retrieve similar documents based on a query
def retrieve(query_text, top_k=5):
    """
    Retrieve the top_k most relevant document chunks from Pinecone for a given query.
    
    :param query_text: The input query for which we want to find relevant chunks
    :param top_k: The number of relevant chunks to retrieve
    :return: A list of retrieval results (most relevant document chunks)
    """
    # Encode the query into an embedding
    query_embedding = hf_model.encode(query_text).tolist()  # Convert to list of floats for Pinecone
    
    # Perform similarity search in Pinecone
    results = index.query(
        vector=query_embedding,  # The query vector (embedding)
        top_k=top_k,  # Number of similar results to return
        include_metadata=True  # Optionally include metadata in the results
    )
    
    return results

# Example usage: Query the Pinecone index
query_text = "What are the benefits of a healthy breakfast?"
retrieved_results = retrieve(query_text)

# Display the retrieved results
print("Retrieved Results:")
for res in retrieved_results['matches']:
    # Safely retrieve the 'content' field and handle missing metadata gracefully
    content = res['metadata'].get('content', 'No content available')
    source = res['metadata'].get('source', 'No source available')
    score = res.get('score', 'No score available')
    
    print(f"Text: {source} | Score: {score} | Content: {content}")

