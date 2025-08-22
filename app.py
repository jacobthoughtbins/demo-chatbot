import streamlit as st
import os
import json
import uuid
import re
from datetime import datetime
from dotenv import load_dotenv
import faiss

# --- LlamaIndex Core Imports ---
from llama_index.core import (
    Settings, StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, Document
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import RecursiveRetriever

# --- Page Configuration & CSS ---
st.set_page_config(
    page_title="Ayurvedic Health Assistant",
    page_icon="🌿",
    layout="wide",
)

def load_css():
    """Loads custom CSS for styling the Streamlit app."""
    st.markdown("""
    <style>
        /* Main App Background */
        .stApp {
            background-color: #121212;
            color: #E0E0E0;
        }
        /* Sidebar Styling */
        .st-emotion-cache-16txtl3 {
            background-color: #1E1E1E;
            border-right: 1px solid #333;
        }
        .st-emotion-cache-16txtl3 h2 {
            color: #4CAF50;
        }
        .st-emotion-cache-16txtl3 .stButton>button {
            width: 100%;
            border-radius: 8px;
            background-color: #4CAF50;
            color: #FFFFFF;
            border: none;
        }
        .st-emotion-cache-16txtl3 .stButton>button:hover {
            background-color: #45a049;
        }
        /* Chat History Buttons & Delete Button */
        .st-emotion-cache-1ja81wh {
            border-radius: 8px;
            background-color: #2C2C2C;
            margin-bottom: 8px;
            transition: background-color 0.3s ease;
        }
        .st-emotion-cache-1ja81wh:hover {
            background-color: #383838;
        }
        .st-emotion-cache-1ja81wh button {
            background-color: transparent;
            color: #E0E0E0;
            border: none;
            text-align: left;
            width: 100%;
        }
        .st-emotion-cache-1ja81wh .st-emotion-cache-1kyxreq {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .st-emotion-cache-1ja81wh .st-emotion-cache-1kyxreq button {
            color: #aaa;
            text-align: center;
        }
        .st-emotion-cache-1ja81wh .st-emotion-cache-1kyxreq button:hover {
            color: #ff4b4b;
        }
        /* Chat Input Styling */
        .st-emotion-cache-13k62yr {
            background-color: #1E1E1E;
            border-top: 1px solid #333;
        }
        .st-emotion-cache-13k62yr textarea {
            background-color: #2C2C2C;
            color: #E0E0E0;
            border: 1px solid #444;
        }
        /* Chat Message Bubbles */
        .stChatMessage {
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .st-emotion-cache-4oy321 { background-color: #264653; } /* User message */
        .st-emotion-cache-janbn0 { background-color: #2A2A2A; } /* Assistant message */
    </style>
    """, unsafe_allow_html=True)

load_css()
load_dotenv()

# Create directories for temporary files if they don't exist
if not os.path.exists("temp_uploads"):
    os.makedirs("temp_uploads")

# --- Backend Logic: Session and Chat Management ---

@st.cache_resource
def load_chat_store():
    """Load chat history from a JSON file or create a new store."""
    if os.path.exists("chat_store.json"):
        return SimpleChatStore.from_persist_path("chat_store.json")
    return SimpleChatStore()

def save_chat_store():
    st.session_state.chat_store.persist(persist_path="chat_store.json")

def load_chat_sessions():
    if os.path.exists("chat_sessions.json"):
        with open("chat_sessions.json", "r") as f:
            return json.load(f)
    return {}

def save_chat_sessions():
    with open("chat_sessions.json", "w") as f:
        json.dump(st.session_state.chat_sessions, f, indent=2)

def create_new_session():
    session_id = str(uuid.uuid4())
    st.session_state.user_doc_index = None
    st.session_state.processed_file_names = []
    return {"id": session_id, "title": "New Chat", "created_at": datetime.now().isoformat()}

def delete_session(session_id_to_delete):
    if session_id_to_delete in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[session_id_to_delete]
        save_chat_sessions()
    st.session_state.chat_store.delete_messages(session_id_to_delete)
    save_chat_store()
    if st.session_state.current_session_id == session_id_to_delete:
        if st.session_state.chat_sessions:
            latest_session = max(st.session_state.chat_sessions.values(), key=lambda s: s['created_at'])
            st.session_state.current_session_id = latest_session['id']
        else:
            new_session = create_new_session()
            st.session_state.chat_sessions[new_session['id']] = new_session
            st.session_state.current_session_id = new_session['id']
            save_chat_sessions()
        st.rerun()

# --- Backend Logic: Model and Index Loading ---

@st.cache_resource
def load_models_and_index():
    """Load embedding model, LLM, and the base FAISS vector index."""
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir="./embedding_cache")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Please set it in your .env file.")
        st.stop()
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=api_key)
    if not os.path.exists("faiss_db"):
        st.error("FAISS database not found. Please ensure the base index has been created.")
        st.stop()
    vector_store = FaissVectorStore.from_persist_dir("faiss_db")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="faiss_db")
    index = load_index_from_storage(storage_context=storage_context, index_id="2a3e044a-5744-41d0-9873-8d679b1571a8")
    return index

# --- Backend Logic: Chat Engine and Response Generation ---

def initialize_chat_engine(session_id: str):
    """Initialize the LlamaIndex chat engine with a robust system prompt."""
    # *** NEW: Enhanced system prompt with a strong medical safety disclaimer ***
    system_prompt = """You are an expert Ayurvedic health assistant. Your knowledge is based on traditional Ayurvedic principles and the context provided.
- Your primary function is to provide information about Ayurveda, health, wellness, and yoga.
- Provide clear, helpful, and direct answers. Be friendly and conversational.
- **IMPORTANT**: Structure your answers clearly using markdown. Use headings and bullet points.
- **CRITICAL**: If a user asks for a cure for a serious disease like cancer, you must state: "Ayurveda can be a supportive therapy for managing symptoms and improving quality of life, but it is not a cure for diseases like cancer. It is essential to consult with a qualified medical doctor for diagnosis and treatment."
- If a question is outside the scope of Ayurveda, health, or the provided documents, politely state: "I am an Ayurvedic health assistant and my knowledge is focused on that area. I can't answer questions on topics like politics, celebrities, or general trivia."
- **MEDICAL SAFETY DISCLAIMER**: If a user uploads a prescription, asks you to validate a dosage, or asks for medical advice, you MUST state that you are an AI assistant and cannot provide medical advice. You can provide general information about herbs from your knowledge base, but you MUST ALWAYS end by strongly advising the user to consult with their qualified practitioner regarding their specific prescription or health condition.
- To keep the conversation interactive, always conclude your response with a friendly, open-ended question.
"""
    memory = ChatMemoryBuffer.from_defaults(token_limit=4000, chat_store=st.session_state.chat_store, chat_store_key=session_id)
    base_retriever = st.session_state.index.as_retriever(similarity_top_k=2)
    if st.session_state.get("user_doc_index"):
        user_retriever = st.session_state.user_doc_index.as_retriever(similarity_top_k=3)
        retriever = RecursiveRetriever("base", retriever_dict={"base": base_retriever, "user_docs": user_retriever})
    else:
        retriever = base_retriever
    return ContextChatEngine.from_defaults(retriever=retriever, memory=memory, system_prompt=system_prompt, llm=Settings.llm)

def update_session_title(first_message: str):
    if st.session_state.chat_sessions[st.session_state.current_session_id]['title'] == "New Chat":
        title = first_message[:40] + '...' if len(first_message) > 40 else first_message
        st.session_state.chat_sessions[st.session_state.current_session_id]['title'] = title
        save_chat_sessions()

# --- Guardrails and Input Validation ---

def is_english(prompt: str) -> bool:
    return not bool(re.search(r'[\u0900-\u097F\u4E00-\u9FFF\uAC00-\uD7AF\u0400-\u04FF]', prompt))

# *** NEW: Function to check if a document is relevant to Ayurveda ***
def is_document_relevant(docs: list[Document]) -> bool:
    """Uses the LLM to classify if the document content is relevant."""
    sample_text = " ".join(doc.get_content() for doc in docs)[:2000] # Get a substantial sample
    
    classification_prompt = f"""
    The following text is from a document uploaded by a user.
    Is this text primarily about Ayurveda, wellness, yoga, natural health, or herbal medicine?
    Answer with a single word: YES or NO.

    --- TEXT SAMPLE ---
    {sample_text}
    --- END SAMPLE ---

    Answer (YES or NO):
    """
    response = Settings.llm.complete(classification_prompt)
    return "YES" in response.text.upper()

def classify_prompt(prompt: str) -> str:
    """Classifies the user's prompt into predefined categories."""
    # (Existing classification logic remains the same)
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "howdy"]
    if prompt.strip().lower() in greetings:
        return "GREETING"
    classification_prompt = f"""
    Classify the user's query into one of the following categories:
    - AYURVEDA_QUERY: A question about Ayurveda, health, wellness, yoga, or a provided document.
    - PERSON_QUERY: A question about a specific named person.
    - GIBBERISH: Nonsensical, random characters, or not a real question.
    - OFF_TOPIC: A question about a topic completely unrelated to Ayurveda.
    Return only the single, most appropriate category name.
    Query: "{prompt}"
    Category:
    """
    response = Settings.llm.complete(classification_prompt)
    category = response.text.strip().upper()
    if "PERSON_QUERY" in category: return "PERSON_QUERY"
    if "AYURVEDA_QUERY" in category: return "AYURVEDA_QUERY"
    if "GIBBERISH" in category: return "GIBBERISH"
    return "OFF_TOPIC"

# --- Streamlit App Initialization ---
if "index" not in st.session_state: st.session_state.index = load_models_and_index()
if "chat_store" not in st.session_state: st.session_state.chat_store = load_chat_store()
if "chat_sessions" not in st.session_state: st.session_state.chat_sessions = load_chat_sessions()
if "user_doc_index" not in st.session_state: st.session_state.user_doc_index = None
if "processed_file_names" not in st.session_state: st.session_state.processed_file_names = []
if "current_session_id" not in st.session_state:
    if st.session_state.chat_sessions:
        st.session_state.current_session_id = max(st.session_state.chat_sessions.values(), key=lambda s: s['created_at'])['id']
    else:
        new_session = create_new_session()
        st.session_state.chat_sessions[new_session['id']] = new_session
        st.session_state.current_session_id = new_session['id']
        save_chat_sessions()
if "chat_engine" not in st.session_state or st.session_state.get("engine_session_id") != st.session_state.current_session_id:
    st.session_state.chat_engine = initialize_chat_engine(st.session_state.current_session_id)
    st.session_state.engine_session_id = st.session_state.current_session_id

# --- UI Rendering: Sidebar for Chat History ---
with st.sidebar:
    st.markdown("## 🌿 Ayurvedic Assistant")
    if st.button("➕ New Chat"):
        new_session = create_new_session()
        st.session_state.chat_sessions[new_session['id']] = new_session
        st.session_state.current_session_id = new_session['id']
        save_chat_sessions()
        st.rerun()
    st.markdown("---")
    st.markdown("### Chat History")
    sorted_sessions = sorted(st.session_state.chat_sessions.values(), key=lambda s: s['created_at'], reverse=True)
    for session in sorted_sessions:
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(session["title"], key=f"session_{session['id']}", use_container_width=True):
                st.session_state.current_session_id = session['id']
                st.session_state.user_doc_index = None
                st.session_state.processed_file_names = []
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{session['id']}", use_container_width=True):
                delete_session(session['id'])

# --- UI Rendering: Main Chat Interface ---
st.header(f"Chat: {st.session_state.chat_sessions.get(st.session_state.current_session_id, {}).get('title', 'New Chat')}")
if st.session_state.current_session_id in st.session_state.chat_store.get_keys():
    for msg in st.session_state.chat_store.get_messages(st.session_state.current_session_id):
        with st.chat_message(msg.role):
            st.markdown(msg.content)

# --- UI Rendering: File Uploader and Parsing Logic ---
uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT) to ask questions about them", type=['pdf', 'docx', 'txt'], accept_multiple_files=True, key=f"uploader_{st.session_state.current_session_id}")

if uploaded_files:
    new_file_names = [f.name for f in uploaded_files]
    if set(new_file_names) != set(st.session_state.get("processed_file_names", [])):
        with st.spinner("Analyzing documents..."):
            file_paths, all_docs = [], []
            for uploaded_file in uploaded_files:
                try:
                    file_path = os.path.join("temp_uploads", uploaded_file.name)
                    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                    all_docs.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
                except Exception as e:
                    st.error(f"Error reading file {uploaded_file.name}: {e}")
            
            if all_docs:
                # *** NEW: Check document relevance before indexing ***
                if is_document_relevant(all_docs):
                    st.success("Document is relevant. Indexing content...")
                    d = 384 # Embedding model dimension
                    faiss_index = faiss.IndexFlatL2(d)
                    vector_store = FaissVectorStore(faiss_index=faiss_index)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    st.session_state.user_doc_index = VectorStoreIndex.from_documents(all_docs, storage_context=storage_context)
                    st.session_state.chat_engine = initialize_chat_engine(st.session_state.current_session_id)
                    st.session_state.processed_file_names = new_file_names
                    st.success(f"Successfully analyzed: **{', '.join(new_file_names)}**.")
                else:
                    st.error("The uploaded document does not seem to be related to Ayurveda or wellness. Please upload a relevant document.")
                    st.session_state.user_doc_index = None # Ensure no index is used
                    st.session_state.processed_file_names = []

            for path in file_paths:
                if os.path.exists(path): os.remove(path)

# --- Main Application Logic: Chat Input and Response ---
if prompt := st.chat_input("Ask about Ayurvedic wellness..."):
    with st.chat_message("user"): st.markdown(prompt)
    try:
        is_first_message = not st.session_state.chat_store.get_messages(st.session_state.current_session_id)
    except KeyError:
        is_first_message = True

    if not is_english(prompt):
        response_text = "I can only converse in English. Please ask your question in English."
        with st.chat_message("assistant"): st.markdown(response_text)
        st.session_state.chat_store.add_message(st.session_state.current_session_id, ChatMessage(role="user", content=prompt))
        st.session_state.chat_store.add_message(st.session_state.current_session_id, ChatMessage(role="assistant", content=response_text))
    else:
        prompt_category = classify_prompt(prompt)
        response_text = ""
        if prompt_category == "GIBBERISH": response_text = "I'm sorry, I didn't understand. Could you please rephrase?"
        elif prompt_category == "PERSON_QUERY": response_text = "As an Ayurvedic assistant, I cannot answer questions about specific people. How can I help with your health today?"
        elif prompt_category == "OFF_TOPIC": response_text = "My knowledge is focused on Ayurveda. I can't answer questions on other topics. How can I help with your wellness today?"
        
        if response_text:
            with st.chat_message("assistant"): st.markdown(response_text)
            st.session_state.chat_store.add_message(st.session_state.current_session_id, ChatMessage(role="user", content=prompt))
            st.session_state.chat_store.add_message(st.session_state.current_session_id, ChatMessage(role="assistant", content=response_text))
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    streaming_response = st.session_state.chat_engine.stream_chat(prompt)
                    full_response = st.write_stream(streaming_response.response_gen)
    
    if is_first_message: update_session_title(prompt)
    save_chat_store()
