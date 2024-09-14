from transformers import AutoTokenizer

# Load the tokenizer for the model you're using
tokenizer = AutoTokenizer.from_pretrained("avsolatorio/NoInstruct-small-Embedding-v0", trust_remote_code=True)

# Define the context text
context = """
You are a helpful physics assistant who answers questions using only the provided context.
Focus on providing clear and accurate physics explanations.
"""

# Tokenize the context and count the tokens
tokens = tokenizer.encode(context)
num_tokens = len(tokens)
print(f"Number of tokens: {num_tokens}")