import gradio as gr
import requests

# Define your backend API URL for chatbot responses
BACKEND_API_URL = "http://localhost:8080/v1/chat/completions"

# Function to call the backend API and get the chatbot response
def chatbot_response(user_message, chat_history):
    headers = {'Content-Type': 'application/json'}
    
    # Construct messages from chat history for the API
    messages = [{"role": "user", "content": msg[0]} if i % 2 == 0 else {"role": "assistant", "content": msg[1]} for i, msg in enumerate(chat_history)]
    
    # Add the new user message to the messages
    messages.append({"role": "user", "content": user_message})
    
    data = {
        'model': '/Users/mustavikhan/Desktop/chatbot/chocolatine-3b-instruct-dpo-revised-q4_k_m.gguf',  # Specify your local model path
        'messages': messages,  # Pass the entire conversation
        'temperature': 0.7,
        'top_k': 40,
        'top_p': 0.4,
        'stop': None
    }
    
    # Make request to the backend API
    response = requests.post(BACKEND_API_URL, json=data)
    
    if response.status_code == 200:
        response_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "Sorry, I couldn't understand that.")
    else:
        response_text = f"Error: {response.status_code}"
    
    # Append the new user message and response to the chat history
    chat_history.append((user_message, response_text))
    
    # Return the updated chat history and clear the input box
    return chat_history, ""

# Gradio interface for the chatbot
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    # Function to clear chat history
    def clear_chat():
        return [], ""

    # Connect input components with functions
    msg.submit(chatbot_response, [msg, chatbot], [chatbot, msg], queue=False)
    clear.click(clear_chat, None, chatbot, queue=False)

# Launch the Gradio app
demo.launch()