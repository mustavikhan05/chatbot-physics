import requests
import time  # To measure time
from retrieval import load_embeddings, load_chunks, build_faiss_index, search_with_continuity, calculate_number_of_chunks
import re

# Define the Llama.cpp server URL
LLAMA_CPP_SERVER_URL = "http://localhost:8080/v1/chat/completions"  # OpenAI-compatible endpoint

# Initialize session context
session_context = []
MAX_TOKENS = 4096  # Define the maximum tokens for your model

def determine_token_count(query):
    base_token_count = 200
    complexity_score = 0
    if re.search(r'\b(explain|detailed|comprehensive|in-depth)\b', query, re.IGNORECASE):
        complexity_score += 300
    if re.search(r'\b(why|how|what)\b', query, re.IGNORECASE):
        complexity_score += 100
    query_length = len(query.split())
    if query_length > 15:
        complexity_score += (query_length - 15) * 10
    return base_token_count + complexity_score

# Function to call Llama.cpp API for generating a response
def call_llama_cpp(prompt, n_predict, llama_server_url=LLAMA_CPP_SERVER_URL):
    headers = {'Content-Type': 'application/json'}
    data = {
        'model': '/Users/mustavikhan/Desktop/chatbot/chocolatine-3b-instruct-dpo-revised-q4_k_m.gguf',  # Specify your local model path
        'messages': session_context + [{"role": "user", "content": prompt}],  # Include session context
        'temperature': 0.7,
        'top_k': 40,
        'top_p': 0.4,
        'max_tokens': n_predict,
        'n_predict': n_predict,
        'stop': None
    }

    try:
        response = requests.post(llama_server_url, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json['choices'][0]['message']['content']
            return response_text
        else:
            return f"Error: {response.status_code}, {response.text}"

    except Exception as e:
        return f"Failed to connect to Llama.cpp server: {str(e)}"

# Function to format the prompt with the retrieved chunks
def format_prompt(query, chunks):
    context = "\n".join(chunks)
    prompt = f"User Query: {query}\n\nContext(do not refer to this context in your response, just answer the question):\n{context}\n\nResponse:"
    return prompt

# Function to calculate the token count of the session context
def calculate_session_token_count():
    total_tokens = 0
    for message in session_context:
        total_tokens += len(message['content'].split())  # Simple word count as a proxy for token count
    return total_tokens

# Function to trim session context if it exceeds the token limit
def trim_session_context():
    global session_context
    while calculate_session_token_count() >= (MAX_TOKENS - 500):  # Leave some buffer space for new responses
        # Remove the oldest message (user-assistant pairs)
        session_context = session_context[2:]

# Main function to handle the query, retrieval, and generation
def generate_response(query, embedding_file, chunk_folder):
    try:
        print("Loading embeddings and chunks...")
        start_time = time.time()
        embeddings = load_embeddings(embedding_file)
        chunks = load_chunks(chunk_folder)

        print("Building FAISS index...")
        index = build_faiss_index(embeddings)

        num_chunks = calculate_number_of_chunks(query)
        print("Searching for relevant chunks...")
        top_chunks = search_with_continuity(query, index, chunks, max_chunks=num_chunks)

        print("Formatting prompt...")
        prompt = format_prompt(query, top_chunks)
        print(f"\nFormatted Prompt:\n{prompt}\n")

        # Check and trim session context if necessary
        trim_session_context()

        n_predict = determine_token_count(query)
        print(f"Selected number of tokens to predict: {n_predict}")

        print("Generating response from Llama.cpp...")
        llama_response = call_llama_cpp(prompt, n_predict=n_predict)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTime taken to generate response: {elapsed_time:.2f} seconds")

        return llama_response

    except Exception as e:
        return f"Error during response generation: {str(e)}"

# Function to clear the session context
def clear_session():
    global session_context
    session_context = []

# Function for the terminal-based chatbot
def run_terminal_chatbot(embedding_file, chunk_folder):
    print("Welcome to the Physics Chatbot. Type 'exit' to quit or 'clear' to reset the session.")
    
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Exiting the chatbot. Goodbye!")
            break
        elif user_input.lower() == "clear":
            clear_session()
            print("Session cleared.")
            continue

        # Generate response
        response = generate_response(user_input, embedding_file, chunk_folder)

        # Store the user query and response in the session context
        session_context.append({"role": "user", "content": user_input})
        session_context.append({"role": "assistant", "content": response})

        # Output the response
        print(f"\nChatbot: {response}\n")

if __name__ == "__main__":
    # Paths to embeddings and chunk files
    embedding_file = "/Users/mustavikhan/Desktop/chatbot/data/embeddings.npy"
    chunk_folder = "/Users/mustavikhan/Desktop/chatbot/data/chunks"
    
    # Run the terminal-based chatbot
    run_terminal_chatbot(embedding_file, chunk_folder)