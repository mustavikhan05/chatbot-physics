import os
import re
import faiss
import numpy as np
import time  # To track time
from transformers import AutoTokenizer, AutoModel
import torch
torch.set_num_threads(1)

# Set the environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the model and tokenizer only once during initialization
model_name = "avsolatorio/NoInstruct-small-Embedding-v0"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

def encode_text(text):
    """
    Encode a single piece of text using the Hugging Face model and tokenizer.
    Handles truncation and token limit.
    """
    tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    # Mean pooling over the token embeddings to get the final representation
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def load_embeddings(embedding_file):
    """
    Load the embeddings from a saved NumPy file.
    """
    start_time = time.time()  # Start timer
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embedding file '{embedding_file}' not found.")
    
    try:
        embeddings = np.load(embedding_file)
    except Exception as e:
        raise ValueError(f"Failed to load embeddings: {str(e)}")
    
    end_time = time.time()  # End timer
    print(f"Time to load embeddings: {end_time - start_time:.2f} seconds")
    
    return embeddings

def build_faiss_index(embeddings):
    """
    Build a FAISS index for fast similarity search.
    """
    start_time = time.time()  # Start timer
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings are empty or None.")
    
    dimension = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric
    index.add(embeddings)  # Add the embeddings to the index
    end_time = time.time()  # End timer
    print(f"Time to build FAISS index: {end_time - start_time:.2f} seconds")
    
    return index

def calculate_number_of_chunks(query, base_chunks=3, max_chunks=5):
    """
    Calculate the number of chunks based on the complexity of the query.
    """
    # Start with the base number of chunks
    complexity_score = 0
    
    # Check for specific keywords or patterns that indicate complexity
    if re.search(r'\b(explain|detailed|comprehensive|in-depth)\b', query, re.IGNORECASE):
        complexity_score += 1  # Add 1 chunk for detailed explanations
    
    if re.search(r'\b(why|how|what)\b', query, re.IGNORECASE):
        complexity_score += 1  # Add 1 chunk for open-ended questions
    
    # Adjust complexity score based on the length of the query
    query_length = len(query.split())
    if query_length > 15:
        complexity_score += (query_length - 15) // 5  # Add chunks for longer queries
    
    # Calculate the total number of chunks, ensuring it stays within the base and max range
    total_chunks = base_chunks + complexity_score
    return min(total_chunks, max_chunks)

def search_with_continuity(query, index, chunks, max_chunks=5):
    """
    Search for the most relevant chunks and ensure that we retrieve adjacent chunks
    to maintain the context continuity.
    """
    start_time = time.time()  # Start timer
    if not query:
        raise ValueError("Query is empty.")
    
    # Encode the query using the modified Hugging Face model
    print("Encoding query...")
    query_embedding = encode_text(query)
    
    # Check if the query embedding is valid
    if query_embedding is None or len(query_embedding) == 0:
        raise ValueError("Failed to generate query embedding.")
    
    # Ensure the query embedding is 2D for FAISS
    query_embedding = query_embedding.reshape(1, -1)  # FAISS expects 2D array (1, embedding_size)
    
    # Search the FAISS index
    print("Performing FAISS search...")
    distances, indices = index.search(query_embedding, max_chunks * 2)  # Retrieve more chunks than needed
    
    # Retrieve the top-k chunks
    selected_chunks = []
    selected_indices = indices[0][:max_chunks]  # Get the top-k indices

    # Retrieve adjacent chunks for context continuity
    for i in selected_indices:
        # Append current chunk
        selected_chunks.append(chunks[i])
        
        # Append preceding chunk if it exists
        if i > 0 and len(selected_chunks) < max_chunks:
            selected_chunks.insert(0, chunks[i - 1])
        
        # Append following chunk if it exists
        if i + 1 < len(chunks) and len(selected_chunks) < max_chunks:
            selected_chunks.append(chunks[i + 1])

    # Limit the final result to max_chunks to avoid exceeding the context limit
    selected_chunks = selected_chunks[:max_chunks]
    
    end_time = time.time()  # End timer
    print(f"Time to search chunks: {end_time - start_time:.2f} seconds")
    
    return selected_chunks

def load_chunks(chunk_folder):
    """
    Load chunks from the folder where text files are stored.
    """
    start_time = time.time()  # Start timer
    if not os.path.exists(chunk_folder):
        raise FileNotFoundError(f"Chunk folder '{chunk_folder}' not found.")
    
    chunks = []
    for file in sorted(os.listdir(chunk_folder)):
        file_path = os.path.join(chunk_folder, file)
        if os.path.isfile(file_path) and file.endswith('.txt'):
            with open(file_path, 'r') as f:
                chunks.append(f.read())
    
    end_time = time.time()  # End timer
    print(f"Time to load chunks: {end_time - start_time:.2f} seconds")
    
    return chunks

if __name__ == "__main__":
    # Paths
    embedding_file = "/Users/mustavikhan/Desktop/chatbot/data/embeddings.npy"
    chunk_folder = "/Users/mustavikhan/Desktop/chatbot/data/chunks"
    
    try:
        # Load embeddings and chunks
        embeddings = load_embeddings(embedding_file)
        chunks = load_chunks(chunk_folder)

        # Build FAISS index
        index = build_faiss_index(embeddings)

        # Example query
        query = input("Enter your query: ")
        num_chunks = calculate_number_of_chunks(query)  # Dynamically calculate the number of chunks
        top_chunks = search_with_continuity(query, index, chunks, max_chunks=num_chunks)

        # Display top results
        print("\nTop results:")
        for i, chunk in enumerate(top_chunks):
            print(f"\nResult {i+1}:\n{chunk}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
