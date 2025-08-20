import streamlit as st
import os
import json
import uuid
import re
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
    """Loads custom CSS for styling the Streamlit app."""
    st.markdown("""
    <style>
        /* Main App Background */
        .stApp {
            background-color: #121212;
            color: #E0E0E0;
        }

        /* --- (other styles remain the same) --- */

        /* Chat Message Bubbles - STABLE VERSION */
        .stChatMessage {
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        /* Target the user message bubble using a stable attribute */
        .stChatMessage:has([data-testid="stChatMessageContent-user"]) {
            background-color: #264653;
        }

        /* Target the assistant message bubble using a stable attribute */
        .stChatMessage:has([data-testid="stChatMessageContent-assistant"]) {
            background-color: #2A2A2A;
        }

    </style>
    """, unsafe_allow_html=True)

load_css()
load_dotenv()

# Create a directory for temporary file uploads if it doesn't exist
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
    """Save the current chat history to a JSON file."""
    st.session_state.chat_store.persist(persist_path="chat_store.json")

def load_chat_sessions():
    """Load session metadata from a JSON file."""
    if os.path.exists("chat_sessions.json"):
        with open("chat_sessions.json", "r") as f:
            return json.load(f)
    return {}

def save_chat_sessions():
    """Save session metadata to a JSON file."""
    with open("chat_sessions.json", "w") as f:
        json.dump(st.session_state.chat_sessions, f, indent=2)

def create_new_session():
    """Create a new chat session and reset relevant session state variables."""
    session_id = str(uuid.uuid4())
    # Reset session-specific states for documents
    st.session_state.user_doc_index = None
    st.session_state.processed_file_names = []
    return {"id": session_id, "title": "New Chat", "created_at": datetime.now().isoformat()}

def delete_session(session_id_to_delete):
    """Delete a chat session and its associated messages."""
    if session_id_to_delete in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[session_id_to_delete]
        save_chat_sessions()

    st.session_state.chat_store.delete_messages(session_id_to_delete)
    save_chat_store()

    # If the deleted session was the active one, switch to the most recent or a new one
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
    system_prompt = """You are an expert Ayurvedic health assistant. Your knowledge is based on traditional Ayurvedic principles and the context provided.
- Your primary function is to provide information about Ayurveda, health, wellness, and yoga.
- Provide clear, helpful, and direct answers. Be friendly and conversational.
- **IMPORTANT**: Structure your answers clearly using markdown. Use headings (e.g., '## Dietary Remedies') and bullet points to organize information.
- **CRITICAL**: If a user asks for a cure for a serious disease like cancer, you must state: "Ayurveda can be a supportive therapy for managing symptoms and improving quality of life, but it is not a cure for diseases like cancer. It is essential to consult with a qualified medical doctor for diagnosis and treatment."
- If a question is outside the scope of Ayurveda, health, or the provided documents, politely state: "I am an Ayurvedic health assistant and my knowledge is focused on that area. I can't answer questions on topics like politics, celebrities, or general trivia."
- To keep the conversation interactive, always conclude your response with a friendly, open-ended question that invites the user to ask for more details or clarification.
- Do not mention the source of your information unless specifically asked.
"""
    memory = ChatMemoryBuffer.from_defaults(token_limit=8000, chat_store=st.session_state.chat_store, chat_store_key=session_id)
    return ContextChatEngine.from_defaults(
        retriever=st.session_state.index.as_retriever(similarity_top_k=2),
        memory=memory,
        system_prompt=system_prompt,
        llm=Settings.llm
    )

def update_session_title(first_message: str):
    """Generate a concise title for the session based on the first user message."""
    if st.session_state.chat_sessions[st.session_state.current_session_id]['title'] == "New Chat":
        if len(first_message) < 50:
            title = first_message
        else:
            prompt = f"Create a very short, concise title (4-5 words max) for a chat session that begins with this user message: '{first_message}'"
            response = Settings.llm.complete(prompt)
            title = response.text.strip().strip('"')
        
        st.session_state.chat_sessions[st.session_state.current_session_id]['title'] = title
        save_chat_sessions()

def get_response_generator(prompt: str):
    """
    Gets a streaming response from the appropriate LlamaIndex engine.
    Prioritizes user-uploaded documents if they exist for the current session.
    """
    if st.session_state.user_doc_index is not None:
        # If documents are uploaded, create a dedicated query engine for them.
        user_doc_query_engine = st.session_state.user_doc_index.as_query_engine(
            similarity_top_k=5, response_mode="no_text"
        )
        
        user_doc_retrieval = user_doc_query_engine.query(prompt)
        
        # Check if any relevant context was found in the documents
        if not user_doc_retrieval.source_nodes:
            # If nothing is found, create a canned response generator.
            def empty_response_generator():
                yield "I couldn't find any relevant information in the uploaded document(s) to answer your question. Could you please ask something more specific about their content?"
            return empty_response_generator()

        context_from_docs = "\n\n---\n\n".join([node.get_content() for node in user_doc_retrieval.source_nodes])
        history = st.session_state.chat_engine.chat_history
        recent_history = history[-4:] if len(history) > 4 else history
        history_str = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in recent_history])

        # A more focused system prompt for document-based queries
        synthesis_prompt = f"""You are an AI assistant. Your task is to answer questions based ONLY on the context provided from user-uploaded documents.
        - Be direct and factual.
        - If the context doesn't contain the answer, say "I could not find an answer to that in the provided documents."
        - Do not use any prior knowledge.

        Relevant Context from Documents:
        {context_from_docs}

        Conversation History:
        {history_str}

        User's Question: {prompt}
        """
        # Manually add user message to history for context in the next turn
        st.session_state.chat_engine.chat_history.append(ChatMessage(role="user", content=prompt))
        
        # Wrapper to extract text from raw stream chunks
        def text_stream_wrapper(raw_stream):
            for chunk in raw_stream:
                yield chunk.delta

        raw_stream = Settings.llm.stream_complete(synthesis_prompt)
        return text_stream_wrapper(raw_stream)

    else:
        # Default behavior: use the main Ayurvedic context chat engine
        streaming_response = st.session_state.chat_engine.stream_chat(prompt)
        return streaming_response.response_gen

# --- Guardrails and Input Validation ---

def is_english(prompt: str) -> bool:
    """
    A more robust check to see if the prompt contains characters
    from non-Latin scripts, allowing for common symbols and punctuation.
    """
    # This regex checks for characters in common non-Latin scripts.
    # It will return True (is English) if no such characters are found.
    return not bool(re.search(r'[\u0900-\u097F\u4E00-\u9FFF\uAC00-\uD7AF\u0400-\u04FF]', prompt))


def classify_prompt(prompt: str) -> str:
    """
    Classifies the user's prompt into predefined categories for robust handling.
    This now includes a GIBBERISH category for the LLM to detect.
    """
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "howdy"]
    if prompt.strip().lower() in greetings:
        return "GREETING"
        
    # Use the LLM for more nuanced classification
    classification_prompt = f"""
    Classify the user's query into one of the following categories. Follow these rules strictly:
    1. If the query asks for personal details, opinions, or analysis about a specific named person (e.g., a celebrity, public figure), it is ALWAYS a 'PERSON_QUERY', regardless of other keywords.
    2. If the query is a general question about Ayurveda, health, wellness, yoga, or asks to analyze a provided document, it is an 'AYURVEDA_QUERY'.
    3. If the query is nonsensical, random characters, or clearly not a real question (e.g., "ajksdhf akjsd"), classify it as 'GIBBERISH'.
    4. If the query is about a topic completely unrelated to Ayurveda (like politics, general trivia, programming), it is 'OFF_TOPIC'.

    Categories:
    - AYURVEDA_QUERY
    - PERSON_QUERY
    - GIBBERISH
    - OFF_TOPIC

    Return only the single, most appropriate category name.

    Query: "{prompt}"
    Category:
    """
    response = Settings.llm.complete(classification_prompt)
    # Clean up the response to get only the category name
    category = response.text.strip().upper()
    
    if "PERSON_QUERY" in category:
        return "PERSON_QUERY"
    elif "AYURVEDA_QUERY" in category:
        return "AYURVEDA_QUERY"
    elif "GIBBERISH" in category:
        return "GIBBERISH"
    else:
        return "OFF_TOPIC"

# --- Streamlit App Initialization ---
if "index" not in st.session_state: st.session_state.index = load_models_and_index()
if "chat_store" not in st.session_state: st.session_state.chat_store = load_chat_store()
if "chat_sessions" not in st.session_state: st.session_state.chat_sessions = load_chat_sessions()

# Session-specific state initialization
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
                # Reset document state when switching sessions
                st.session_state.user_doc_index = None
                st.session_state.processed_file_names = []
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{session['id']}", use_container_width=True):
                delete_session(session['id'])
                st.rerun()

# --- UI Rendering: Main Chat Interface ---
st.header(f"Chat: {st.session_state.chat_sessions.get(st.session_state.current_session_id, {}).get('title', 'New Chat')}")

# Display chat messages from history
for msg in st.session_state.chat_engine.chat_history:
    with st.chat_message(msg.role):
        st.markdown(msg.content)

# --- UI Rendering: File Uploader and Parsing Logic ---
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT) to ask questions about them", 
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.current_session_id}"
)

if uploaded_files:
    # Check if the uploaded files are new for this session
    new_file_names = [f.name for f in uploaded_files]
    if set(new_file_names) != set(st.session_state.processed_file_names):
        st.session_state.processed_file_names = [] # Reset list for reprocessing
        file_paths = []
        all_docs = []
        
        with st.spinner(f"Analyzing {len(uploaded_files)} document(s)..."):
            for uploaded_file in uploaded_files:
                try:
                    file_path = os.path.join("temp_uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                    
                    # Load data from the single file
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    user_docs = reader.load_data()
                    all_docs.extend(user_docs)
                    st.session_state.processed_file_names.append(uploaded_file.name)

                except Exception as e:
                    # More user-friendly error for corrupted files
                    st.error(f"Sorry, the file '{uploaded_file.name}' seems to be corrupted or unreadable. Please try a different file. Error: {e}")
                    # Clean up the problematic file
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    continue # Move to the next file

            if all_docs:
                # Create a single index from all successfully loaded documents
                st.session_state.user_doc_index = VectorStoreIndex.from_documents(all_docs)
                st.success(f"Successfully analyzed **{', '.join(st.session_state.processed_file_names)}**. You can now ask questions about them.")
            
            # Clean up all temporary files
            for path in file_paths:
                if os.path.exists(path):
                    os.remove(path)

# --- Main Application Logic: Chat Input and Response ---
if prompt := st.chat_input("Ask about Ayurvedic wellness..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    is_first_message = len(st.session_state.chat_engine.chat_history) == 0

    # --- Multi-stage Guardrail Check ---
    if not is_english(prompt):
        response_text = "I am currently able to converse only in English. Please ask your question in English."
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.chat_engine.chat_history.append(ChatMessage(role="user", content=prompt))
        st.session_state.chat_engine.chat_history.append(ChatMessage(role="assistant", content=response_text))
    else:
        prompt_category = classify_prompt(prompt)
        
        if prompt_category == "GIBBERISH":
            response_text = "I'm sorry, I didn't understand your question. Could you please rephrase it?"
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.chat_engine.chat_history.append(ChatMessage(role="user", content=prompt))
            st.session_state.chat_engine.chat_history.append(ChatMessage(role="assistant", content=response_text))
        
        elif prompt_category == "PERSON_QUERY":
            response_text = "As an Ayurvedic health assistant, my expertise is focused on wellness and natural health. I cannot answer questions about specific people. How can I help you with your health today?"
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.chat_engine.chat_history.append(ChatMessage(role="user", content=prompt))
            st.session_state.chat_engine.chat_history.append(ChatMessage(role="assistant", content=response_text))

        elif prompt_category == "OFF_TOPIC":
            response_text = "I am an Ayurvedic health assistant and my knowledge is focused on that area. I can't answer questions on topics like politics, celebrities, or general trivia. How can I help you with your wellness today?"
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.chat_engine.chat_history.append(ChatMessage(role="user", content=prompt))
            st.session_state.chat_history.append(ChatMessage(role="assistant", content=response_text))

        else: # This handles GREETING and AYURVEDA_QUERY
            # --- CORE RESPONSE LOGIC ---
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_generator = get_response_generator(prompt)
                    full_response = st.write_stream(response_generator)


            if st.session_state.user_doc_index is not None:
                st.session_state.chat_engine.chat_history.append(
                    ChatMessage(role="assistant", content=full_response)
                )

    # Update title on the first message, regardless of category.
    if is_first_message:
        update_session_title(prompt)

    # Save chat store after every interaction
    save_chat_store()
