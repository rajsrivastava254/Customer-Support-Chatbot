# --- STEP 1: SQLite Fix (Must be at the very top) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st  
from gtts import gTTS
import io
from dotenv import load_dotenv

# --- STEP 2: Updated 2026 LangChain Imports ---
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# API Key Validation
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
    st.stop()

@st.cache_resource
def setup_rag_chain():
    """
    Loads data, splits it, creates embeddings, stores in a vector DB.
    """
    if not os.path.exists('support_data.txt'):
        with open('support_data.txt', 'w') as f:
            f.write("Our shipping takes 3-5 days. Returns are accepted within 30 days.")
            
    loader = TextLoader('support_data.txt')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings)
    return db.as_retriever()

# --- STEP 3: Chain Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt = ChatPromptTemplate.from_template("""
Answer the user's question based only on the following context:
<context>
{context}
</context>
Question: {input}
""")

# Setup the components
retriever = setup_rag_chain()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- STEP 4: Streamlit UI ---
st.title("📄 Customer Support Chatbot 🗣️")
st.write("Ask about our policies or use the samples below.")

def process_question(query):
    with st.spinner("Finding an answer..."):
        # FIX: Using .invoke() instead of calling the object like a function
        response = retrieval_chain.invoke({"input": query})
        
        # Retrieval chains return a dict; we extract the 'answer' key
        answer = response.get("answer", "I'm sorry, I couldn't find an answer.")
        
        # Audio generation
        audio_io = io.BytesIO()
        tts = gTTS(text=answer, lang='en')
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        return answer, audio_io

# Handle User Input
user_input = st.text_input("Your question:", key="text_input")

# Sample Questions
st.subheader("Sample questions:")
samples = ["What are your shipping options?", "What is your return policy?"]
cols = st.columns(len(samples))

# We'll check if a button was clicked
clicked_sample = None
for i, sample in enumerate(samples):
    if cols[i].button(sample):
        clicked_sample = sample

# Determine final query (prioritize clicked button over text box)
final_query = clicked_sample if clicked_sample else user_input

if final_query:
    answer_text, audio_data = process_question(final_query)
    st.subheader("Answer:")
    st.write(answer_text)
    st.audio(audio_data)