__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    return retriever

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
# --- Part 3: Create a User Interface with Streamlit ---
# ==============================================================================

st.title("📄 Customer Support Chatbot 🗣️")
st.write("Ask me anything about our policies, or try a sample question below.")

# --- Text Input Box ---
user_input = st.text_input("Your question:")

if user_input:
    # When the user enters a question, invoke the RAG chain
    with st.spinner("Finding an answer..."):
        response = retrieval_chain.invoke({"input": user_input})
        answer_text = response["answer"]

        # Generate and display voice response
        audio_io = io.BytesIO()
        tts = gTTS(text=answer_text, lang='en')
        tts.write_to_fp(audio_io)

    # Display the written and spoken answers
    st.subheader("Answer:")
    st.write(answer_text)
    st.audio(audio_io)

# --- Sample Questions Section (New Feature) ---
st.subheader("Or try one of these sample questions:")

sample_questions = [
    "What are your shipping options?",
    "What is your return policy?",
    "Do you ship internationally?",
    "How do I reset my password?"
]

# Display buttons in columns for a cleaner look
col1, col2 = st.columns(2)

for i, question in enumerate(sample_questions):
    if i % 2 == 0:
        with col1:
            if st.button(question, use_container_width=True):
                user_input = question  # Set the user_input to the question
    else:
        with col2:
            if st.button(question, use_container_width=True):
                user_input = question  # Set the user_input to the question

# This part is now outside the original if-statement to handle both text and button inputs
if user_input and st.button not in st.session_state:
    with st.spinner("Finding an answer..."):
        response = retrieval_chain.invoke({"input": user_input})
        answer_text = response["answer"]

        # Generate and display voice response
        audio_io = io.BytesIO()
        tts = gTTS(text=answer_text, lang='en')
        tts.write_to_fp(audio_io)

    st.subheader("Answer:")
    st.write(answer_text)
    st.audio(audio_io)