import streamlit as st
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv

# --- LlamaIndex Core Imports ---
from llama_index.core import (
    Settings, StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage

# --- Page Configuration & CSS ---
st.set_page_config(
    page_title="Ayurvedic Health Assistant",
    page_icon="🌿",
    layout="wide",
)

def load_css():
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
        .st-emotion-cache-1ja81wh { /* Div container for columns */
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
        .st-emotion-cache-1ja81wh .st-emotion-cache-1kyxreq { /* Delete button column */
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

# Create a directory for temporary file uploads
if not os.path.exists("temp_uploads"):
    os.makedirs("temp_uploads")

# --- Backend Logic ---

@st.cache_resource
def load_chat_store():
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

@st.cache_resource
def load_models_and_index():
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir="./embedding_cache")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found.")
        st.stop()
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=api_key)
    if not os.path.exists("faiss_db"):
        st.error("FAISS database not found.")
        st.stop()
    vector_store = FaissVectorStore.from_persist_dir("faiss_db")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="faiss_db")
    index = load_index_from_storage(storage_context=storage_context, index_id="2a3e044a-5744-41d0-9873-8d679b1571a8")
    return index

def initialize_chat_engine(session_id: str):
    system_prompt = """You are an expert Ayurvedic health assistant... (Your full prompt here)"""
    memory = ChatMemoryBuffer.from_defaults(token_limit=15000, chat_store=st.session_state.chat_store, chat_store_key=session_id)
    return ContextChatEngine.from_defaults(retriever=st.session_state.index.as_retriever(similarity_top_k=5), memory=memory, system_prompt=system_prompt, llm=Settings.llm)

def update_session_title(first_message: str):
    if len(first_message) < 50:
        title = first_message
    else:
        prompt = f"Create a very short, concise title (4-5 words max) for a chat session that begins with this user message: '{first_message}'"
        response = Settings.llm.complete(prompt)
        title = response.text.strip().strip('"')
    
    st.session_state.chat_sessions[st.session_state.current_session_id]['title'] = title
    save_chat_sessions()

def get_response(prompt: str):
    # --- NEW MEMORY-AWARE LOGIC ---
    if st.session_state.user_doc_index is not None:
        # 1. Create query engines that only retrieve context, not synthesize
        main_query_engine = st.session_state.index.as_query_engine(similarity_top_k=2, response_mode="no_text")
        user_doc_query_engine = st.session_state.user_doc_index.as_query_engine(similarity_top_k=2, response_mode="no_text")

        # 2. Retrieve context from both sources
        main_retrieval = main_query_engine.query(prompt)
        user_doc_retrieval = user_doc_query_engine.query(prompt)
        
        # 3. Combine the retrieved text into a single context string
        combined_context = "\n\n---\n\n".join([node.get_content() for node in main_retrieval.source_nodes + user_doc_retrieval.source_nodes])

        # 4. Get recent chat history
        history = st.session_state.chat_engine.chat_history
        recent_history = history[-4:] if len(history) > 4 else history
        history_str = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in recent_history])

        # 5. Construct a final, comprehensive prompt for the LLM
        final_prompt = f"""You are an expert Ayurvedic health assistant.
        
        Here is the recent conversation history:
        <history>
        {history_str}
        </history>

        Here is some relevant context retrieved from the user's uploaded document and your general knowledge base:
        <context>
        {combined_context}
        </context>

        Based on the conversation history AND the provided context, answer the user's latest question.
        User's latest question: '{prompt}'
        """
        
        # 6. Generate the final response directly from the LLM
        response = Settings.llm.complete(final_prompt)
        return str(response)
    else:
        # If no document is uploaded, use the standard stateful chat engine
        response = st.session_state.chat_engine.chat(prompt)
        return str(response)

# --- Streamlit App Initialization ---
if "index" not in st.session_state: st.session_state.index = load_models_and_index()
if "chat_store" not in st.session_state: st.session_state.chat_store = load_chat_store()
if "chat_sessions" not in st.session_state: st.session_state.chat_sessions = load_chat_sessions()
if "user_doc_index" not in st.session_state: st.session_state.user_doc_index = None

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

# --- Sidebar for Chat History ---
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
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{session['id']}", use_container_width=True):
                delete_session(session['id'])
                st.rerun()

# --- Main Chat Interface ---
st.header(f"Chat: {st.session_state.chat_sessions[st.session_state.current_session_id]['title']}")

for msg in st.session_state.chat_engine.chat_history:
    with st.chat_message(msg.role):
        st.markdown(msg.content)

uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, TXT) to ask questions about it", 
    type=['pdf', 'docx', 'txt'],
    key=f"uploader_{st.session_state.current_session_id}"
)

if uploaded_file is not None:
    try:
        uploaded_file_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            reader = SimpleDirectoryReader(input_files=[uploaded_file_path])
            user_docs = reader.load_data()
            st.session_state.user_doc_index = VectorStoreIndex.from_documents(user_docs)
        
        st.success(f"Successfully analyzed **{uploaded_file.name}**. You can now ask questions about it.")
        
        if os.path.exists(uploaded_file_path):
            os.remove(uploaded_file_path)
            
    except Exception as e:
        st.error(f"Failed to process file: {e}")
        st.session_state.user_doc_index = None

prompt = st.chat_input("Ask about Ayurvedic wellness...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    is_first_message = len(st.session_state.chat_engine.chat_history) == 0

    with st.spinner("Thinking..."):
        response_text = get_response(prompt)
        
        st.session_state.chat_engine._memory.put(ChatMessage(role="user", content=prompt))
        st.session_state.chat_engine._memory.put(ChatMessage(role="assistant", content=response_text))
        
        with st.chat_message("assistant"):
            st.markdown(response_text)

    if is_first_message:
        update_session_title(prompt)

    save_chat_store()
    st.rerun()
