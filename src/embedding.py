from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os

def load_chunks(chunk_folder):
    """
    Loads the text chunks from the specified folder.
    """
    chunks = []
    for file_name in sorted(os.listdir(chunk_folder)):
        file_path = os.path.join(chunk_folder, file_name)
        with open(file_path, "r") as file:
            chunks.append(file.read())
    return chunks

def generate_embeddings(chunks, model_name="avsolatorio/NoInstruct-small-Embedding-v0", max_tokens=512):
    """
    Generates embeddings for the text chunks using Hugging Face's transformers library.
    Handles truncation based on token limit.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    
    for chunk in chunks:
        # Tokenize the chunk and ensure it does not exceed the max token limit
        tokens = tokenizer(chunk, truncation=True, max_length=max_tokens, return_tensors="pt")
        
        # Generate embeddings
        with torch.no_grad():
            output = model(**tokens)
        
        # Take the mean of the last hidden state (common pooling strategy)
        pooled_output = output.last_hidden_state.mean(dim=1)
        embeddings.append(pooled_output.squeeze().numpy())
    
    return embeddings

def save_embeddings(embeddings, output_file):
    """
    Saves embeddings as a numpy file.
    """
    np.save(output_file, embeddings)

if __name__ == "__main__":
    chunk_folder = "/Users/mustavikhan/Desktop/chatbot/data/chunks"
    output_file = "/Users/mustavikhan/Desktop/chatbot/data/embeddings.npy"

    # Load the chunks
    chunks = load_chunks(chunk_folder)

    # Generate embeddings for each chunk with truncation handling
    embeddings = generate_embeddings(chunks)

    # Save embeddings to a file
    save_embeddings(embeddings, output_file)

    print(f"Embeddings saved to {output_file}")