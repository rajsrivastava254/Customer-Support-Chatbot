# AI Customer Support Chatbot with RAG

A conversational AI chatbot designed to provide instant and accurate answers to customer support questions using a custom knowledge base.
This project leverages Retrieval-Augmented Generation (RAG) to ensure responses are factual and grounded in provided documents, minimizing AI hallucinations.

**🚀 Overview**

This application provides a user-friendly web interface where users can ask questions in natural language.
The backend processes the query using a RAG pipeline:

Retrieves the most relevant information from a local knowledge base (support_data.txt).

Uses Google Gemini Pro to generate a coherent, context-aware answer.

Additionally, the chatbot includes a Text-to-Speech (TTS) feature to speak answers aloud, improving accessibility.

**✨ Key Features**

Retrieval-Augmented Generation (RAG): Ensures answers are accurate and based solely on the provided knowledge base.

Free & Local Embeddings: Powered by Hugging Face models for cost-free local embeddings.

Google Gemini LLM: Leverages the free tier of Gemini Pro for high-quality responses.

Interactive UI: Built with Streamlit for a simple, intuitive experience.

Text-to-Speech (TTS): Converts answers to audio using gTTS.

Customizable Knowledge: Easily update chatbot knowledge by editing a single file (support_data.txt).

**🛠️ Technologies Used**

Backend: Python

AI Framework: LangChain

LLM: Google Gemini

Embeddings: Hugging Face Transformers (all-MiniLM-L6-v2)

Vector Database: ChromaDB

Web Interface: Streamlit

Text-to-Speech: gTTS

**⚙️ Setup & Installation**

Follow these steps to set up the project on your local machine.

1. Clone the Repository
git clone https://github.com/rajsrivastava254/Customer-Support-Chatbot.git
cd Customer-Support-Chatbot

2. Create a Virtual Environment (Recommended)
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Set Up API Keys

Create a .env file in the root directory and add your Google Gemini API key:

GOOGLE_API_KEY="your-google-api-key-goes-here"

▶️ How to Run

Run the following command to start the chatbot:

streamlit run chatbot.py


A new browser tab will open with the chatbot interface.

📝 How to Customize

To update the chatbot’s knowledge base:

Open the support_data.txt file.

Add, remove, or modify Q&A pairs.

Restart the chatbot, and it will use the updated information.

📌 Future Improvements

Add multi-file knowledge base support (PDF, CSV, Docs).

Enable multilingual Q&A.

Enhance UI with chat history and user profiles.
