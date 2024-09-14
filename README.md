Here’s a documentation guide for setting up this GitHub repository locally, including how to install Llama.cpp on Windows, download the model, and run the model in the command line using Llama.cpp. Follow the steps below for the complete setup:

---

# Local Setup Guide for this GitHub Repository

## 1. Install Llama.cpp on Windows

Llama.cpp is required to run the model on your local machine. Follow these steps to install it on Windows:

### Steps to Install:
1. Go to the [Llama.cpp GitHub Repository](https://github.com/ggerganov/llama.cpp).
2. Follow the installation instructions for Windows:
   - Clone the Llama.cpp repository:
     ```bash
     git clone https://github.com/ggerganov/llama.cpp
     ```
   - Navigate to the repository folder:
     ```bash
     cd llama.cpp
     ```
   - Install dependencies (e.g., `cmake`, `git`, `g++`).
   - Build the project:
     ```bash
     cmake .
     cmake --build . --config Release
     ```

   For more details on installation, check the Llama.cpp repository.

---

## 2. Download the Model

You will need the Chocolatine-3B-Instruct-DPO-Revised-Q4_K_M-GGUF model for running inference. Download it from the following link:

- [Download the model from Hugging Face](https://huggingface.co/jpacifico/Chocolatine-3B-Instruct-DPO-Revised-Q4_K_M-GGUF/resolve/main/README.md?download=true).

Save the model file in an accessible directory on your machine. Make sure to note the path where the model is saved for use in the following steps.

---

## 3. Install Required Dependencies

To set up the necessary environment and dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/chatbot-physics.git
   ```
   Replace `your-repo` with the correct repository name.

2. Navigate to the repository directory:
   ```bash
   cd chatbot-physics
   ```

3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

4. Install the required dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Dry Run the Model in the Command Line Using Llama.cpp

Once Llama.cpp and the model are set up, you can perform a dry run of the model using the command line. Use the following command to run the model in Llama.cpp:

1. Ensure you are in the `llama.cpp` directory.

2. Run the model:
   ```bash
   ./main -m /path/to/model/file -p "Hello, how can I assist you?"
   ```
   Replace `/path/to/model/file` with the actual path to the downloaded Chocolatine-3B model.

   The `-p` flag is used to pass the prompt to the model, and you can modify the prompt as needed.

---

## 5. Clone This Repository

To clone this repository, use the following command:

1. Open a terminal or command prompt.

2. Run the following Git command:
   ```bash
   git clone https://github.com/your-repo/chatbot-physics.git
   ```

3. Navigate into the project directory:
   ```bash
   cd chatbot-physics
   ```

---

Follow the steps above, and you should be able to successfully set up the project locally on your machine. For any additional help or troubleshooting, refer to the respective documentation links provided above.