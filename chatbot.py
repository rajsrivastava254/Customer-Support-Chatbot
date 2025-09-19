# ==============================================================================
# --- Part 1: Setup and Data Preparation (Steps 1-3) ---
# ==============================================================================
import os
import streamlit as st
from gtts import gTTS
import io
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Ensure the OpenAI API key is set
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Make sure it's set in your .env file.")

# Function to setup the RAG components (to avoid re-running on every interaction)
# COPY AND PASTE THIS ENTIRE FUNCTION INTO YOUR chatbot.py FILE

@st.cache_resource
def setup_rag_chain():
    """
    Loads data, splits it, creates embeddings, stores in a vector DB,
    and sets up the retrieval chain.
    """
    # 1. Load your data
    loader = TextLoader('support_data.txt')
    documents = loader.load()

    # 2. Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # 3. Create embeddings and store in vector DB using the FREE model
    #    This is the most important change.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    return retriever

# ==============================================================================
# --- Part 2: Building the Question-Answering Chain (Step 4) ---
# ==============================================================================

# The LLM to use for generating answers
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# The prompt template guides the LLM on how to answer
prompt = ChatPromptTemplate.from_template("""
Answer the user's question based only on the following context:

<context>
{context}
</context>

Question: {input}
""")

# This chain will combine the retrieved documents into the prompt
document_chain = create_stuff_documents_chain(llm, prompt)

# Setup the retriever once
retriever = setup_rag_chain()

# This is the final RAG chain. It retrieves documents AND then generates an answer.
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# ==============================================================================
# --- Part 3: Create a User Interface with Streamlit (Step 5) ---
# ==============================================================================

st.title("📄 Customer Support Chatbot 🗣️")
st.write("Ask me anything about our shipping, returns, or account policies!")

# Get user input from a text box
user_input = st.text_input("Your question:")

if user_input:
    # When the user enters a question, invoke the RAG chain
    with st.spinner("Finding an answer..."):
        response = retrieval_chain.invoke({"input": user_input})
        answer_text = response["answer"]

        # --- New Voice Generation Code ---
        # Create an in-memory audio file
        audio_io = io.BytesIO()
        # Use gTTS to convert the text answer to speech
        tts = gTTS(text=answer_text, lang='en')
        # Write the audio data to the in-memory file
        tts.write_to_fp(audio_io)
        # --- End of New Code ---

    # Display the written answer
    st.subheader("Answer:")
    st.write(answer_text)

    # Display the audio player for the spoken answer
    st.subheader("Spoken Answer:")
    st.audio(audio_io)