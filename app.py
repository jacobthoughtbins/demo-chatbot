import streamlit as st
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.core.prompts import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import html
import textwrap

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Ayurvedic Health Assistant",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ChatGPT-style CSS with delete functionality
st.markdown("""
<style>
    :root {
        --brand-green: #19c37d;
        --brand-green-dark: #065f46;
        --brand-gradient: linear-gradient(135deg, #0ea37f 0%, #065f46 100%);

        /* Peach theme variables */
        --peach-bg: #f7eee7;         /* page background */
        --peach-bg-2: #f3e5db;       /* panels/input background */
        --peach-text: #2f2a26;       /* main text color */
        --peach-muted: #7b6e66;      /* secondary text */
        --peach-border: #e4cfc2;     /* soft border */
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Hide any potential invisible elements that might take space */
    .stToolbar {display: none !important;}
    .stDecoration {display: none !important;}
    .stActionButton {display: none !important;}

    /* Remove any default Streamlit spacing */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Ensure all Streamlit containers are properly sized */
    .stMarkdown, .stMarkdown > div {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Remove all default padding and margins */
    .main > div {
        padding: 0 !important;
        margin: 0 !important;
        height: 100% !important;
        overflow: visible !important;
    }

    /* Ensure all child containers allow scrolling */
    .main .element-container,
    .main .stMarkdown,
    .main .stMarkdown > div {
        overflow: visible !important;
        height: auto !important;
        max-height: none !important;
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* Ensure the main content container takes full height */
    .main > div:first-child {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: none !important;
        position: relative !important;
        overflow: visible !important;
        width: 100% !important;
        display: block !important;
    }

    /* Ensure main content area is properly positioned */
    .main {
        position: relative !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
        width: 100% !important;
        display: block !important;
    }

    /* Full height app */
    .stApp {
        height: 100vh !important;
        overflow: visible !important;
        position: static !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Ensure root container has no margins */
    .stApp > div {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        height: 100% !important;
    }

    /* Sidebar styling - ChatGPT style */
    section[data-testid="stSidebar"] {
        width: 260px !important;
        background: #171717 !important;
        border-right: 1px solid #2d2d2d !important;
        height: 100vh !important;
        overflow: hidden !important;
    }

    section[data-testid="stSidebar"] > div {
        padding: 12px !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* New Chat Button */
    .new-chat-btn {
        background: transparent !important;
        border: 1px solid #4d4d4f !important;
        color: #ececf1 !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
        margin-bottom: 16px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    .new-chat-btn:hover {
        background: #2d2d2d !important;
        border-color: #565869 !important;
    }

    /* Chat History */
    .chat-history {
        flex: 1 !important;
        overflow-y: auto !important;
        margin-top: 8px !important;
    }

    /* Chat history item container with hover effect for delete button */
    .chat-history-item-container {
        position: relative !important;
        margin-bottom: 4px !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }

    .chat-history-item-container:hover {
        background: #2d2d2d !important;
    }

    .chat-history-item-container:hover .delete-btn {
        opacity: 1 !important;
        visibility: visible !important;
    }

    .chat-history-item {
        background: transparent !important;
        border: none !important;
        color: #ececf1 !important;
        padding: 12px 40px 12px 16px !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        text-align: left !important;
        display: block !important;
        word-wrap: break-word !important;
        line-height: 1.4 !important;
        position: relative !important;
    }

    .chat-history-item.active {
        background: #343541 !important;
    }

    .chat-history-date {
        font-size: 11px !important;
        color: #8e8ea0 !important;
        margin-top: 4px !important;
    }

    /* Delete button */
    .delete-btn {
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        background: #ef4444 !important;
        border: none !important;
        border-radius: 4px !important;
        color: white !important;
        width: 24px !important;
        height: 24px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        opacity: 0 !important;
        visibility: hidden !important;
        transition: all 0.2s ease !important;
        font-size: 12px !important;
        z-index: 10 !important;
    }

    .delete-btn:hover {
        background: #dc2626 !important;
        transform: translateY(-50%) scale(1.1) !important;
    }

    /* Main chat container - Centered layout */
    .chat-container {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: flex-start !important;
        background: var(--peach-bg) !important;
        position: relative !important; /* let Streamlit layout handle sidebar width */
        min-height: 100vh !important;
        padding: 20px !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }

    /* Chat box - Centered container */
    .chat-box {
        width: 100% !important;
        max-width: 800px !important;
        height: calc(100vh - 40px) !important; /* viewport height minus container padding */
        max-height: calc(100vh - 40px) !important;
        background: var(--peach-bg-2) !important;
        border-radius: 12px !important;
        border: 1px solid var(--peach-border) !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        min-height: 0 !important; /* allow children to size */
    }

    /* Chat messages area - Scrollable inside container */
    .messages-container {
        flex: 1 1 auto !important; /* grow and shrink within chat-box */
        min-height: 0 !important; /* critical for flex scrolling */
        height: auto !important;
        max-height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: 16px !important;
        background: transparent !important;
        width: 100% !important;
        box-sizing: border-box !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-gutter: stable both-edges !important;
    }

    /* Individual message wrapper */
    .message-wrapper {
        width: 100% !important;
        margin-bottom: 16px !important;
        display: flex !important;
        align-items: flex-start !important;
        gap: 12px !important;
        box-sizing: border-box !important;
        min-height: auto !important;
        height: auto !important;
        flex-shrink: 0 !important;
        overflow: visible !important;
    }

    .message-wrapper.user {
        background: transparent !important;
    }

    .message-wrapper.assistant {
        background: transparent !important;
    }

    /* Message avatar */
    .message-avatar {
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 16px !important;
        flex-shrink: 0 !important;
        margin-top: 2px !important;
    }

    .user-avatar {
        background: #5436da !important;
    }

    .assistant-avatar {
        background: #10a37f !important;
    }

    .message-avatar {
        width: 32px !important;
        height: 32px !important;
        border-radius: 4px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 16px !important;
        flex-shrink: 0 !important;
    }

    .user-avatar {
        background: #5436da !important;
        color: white !important;
    }

    .assistant-avatar {
        background: #19c37d !important;
        color: white !important;
    }

    .message-text {
        flex: 1 !important;
        color: var(--peach-text) !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        margin: 0 !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: pre-wrap !important;
        max-width: 100% !important;
        overflow: visible !important;
        height: auto !important;
        min-height: auto !important;
        background: var(--peach-bg) !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        border: 1px solid var(--peach-border) !important;
    }

    /* Chat input area - Bottom of container */
    .chat-input-container {
        background: var(--peach-bg-2) !important;
        padding: 16px !important;
        border-top: 1px solid var(--peach-border) !important;
        flex-shrink: 0 !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        box-sizing: border-box !important;
        border-radius: 0 0 12px 12px !important;
    }

    .chat-input-wrapper {
        max-width: 768px !important;
        margin: 0 auto !important;
        padding: 0 24px !important;
        position: relative !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Style the actual Streamlit chat input */
    .stChatInputContainer {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .stChatInputContainer > div {
        background: var(--peach-bg) !important;
        border: 1px solid var(--peach-border) !important;
        border-radius: 12px !important;
        padding: 0 !important;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1) !important;
    }

    .stChatInputContainer textarea {
        background: transparent !important;
        color: var(--peach-text) !important;
        border: none !important;
        font-size: 16px !important;
        padding: 12px 48px 12px 16px !important;
        resize: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    .stChatInputContainer textarea::placeholder {
        color: var(--peach-muted) !important;
    }

    /* Send button styling */
    .stChatInputContainer button {
        background: #19c37d !important;
        border: none !important;
        border-radius: 6px !important;
        color: white !important;
        padding: 8px !important;
        margin: 6px !important;
        cursor: pointer !important;
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
    }

    /* Welcome message styling */
    .welcome-container {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 32px 24px !important;
        text-align: center !important;
        width: 100% !important;
        box-sizing: border-box !important;
        max-width: 100% !important;
        margin: 0 auto !important;
    }

    .welcome-title {
        color: #ececf1 !important;
        font-size: 32px !important;
        font-weight: 600 !important;
        margin-bottom: 16px !important;
        text-align: center !important;
        width: 100% !important;
    }

    .welcome-subtitle {
        color: #8e8ea0 !important;
        font-size: 16px !important;
        margin-bottom: 32px !important;
        max-width: 600px !important;
        line-height: 1.5 !important;
        margin-left: auto !important;
        margin-right: auto !important;
        text-align: center !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Confirmation dialog styling */
    .delete-confirmation {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        background: rgba(0, 0, 0, 0.5) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        z-index: 1000 !important;
    }

    .delete-dialog {
        background: #2d2d2d !important;
        border-radius: 8px !important;
        padding: 24px !important;
        max-width: 400px !important;
        width: 90% !important;
        color: #ececf1 !important;
    }

    .delete-dialog h3 {
        margin: 0 0 16px 0 !important;
        color: #ececf1 !important;
    }

    .delete-dialog p {
        margin: 0 0 24px 0 !important;
        color: #8e8ea0 !important;
    }

    .delete-dialog-buttons {
        display: flex !important;
        gap: 12px !important;
        justify-content: flex-end !important;
    }

    .delete-dialog-btn {
        padding: 8px 16px !important;
        border-radius: 6px !important;
        border: none !important;
        cursor: pointer !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    .delete-dialog-btn.cancel {
        background: #4d4d4f !important;
        color: #ececf1 !important;
    }

    .delete-dialog-btn.confirm {
        background: #ef4444 !important;
        color: white !important;
    }

    .delete-dialog-btn:hover {
        opacity: 0.8 !important;
    }

    /* Scrollbar styling */
    .messages-container::-webkit-scrollbar,
    .chat-history::-webkit-scrollbar {
        width: 8px !important;
    }

    .messages-container::-webkit-scrollbar-track,
    .chat-history::-webkit-scrollbar-track {
        background: transparent !important;
    }

    .messages-container::-webkit-scrollbar-thumb,
    .chat-history::-webkit-scrollbar-thumb {
        background: #565869 !important;
        border-radius: 4px !important;
    }

    .messages-container::-webkit-scrollbar-thumb:hover,
    .chat-history::-webkit-scrollbar-thumb:hover {
        background: #6b7280 !important;
    }

    /* Hide Streamlit chat message containers */
    .stChatMessage {
        display: none !important;
    }

    /* Loading animation */
    .typing-indicator {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        color: #8e8ea0 !important;
        font-style: italic !important;
    }

    .typing-dots {
        display: flex !important;
        gap: 4px !important;
    }

    .typing-dots span {
        width: 6px !important;
        height: 6px !important;
        background: #8e8ea0 !important;
        border-radius: 50% !important;
        animation: typing 1.4s infinite !important;
    }

    .typing-dots span:nth-child(2) {
        animation-delay: 0.2s !important;
    }

    .typing-dots span:nth-child(3) {
        animation-delay: 0.4s !important;
    }

    @keyframes typing {
        0%, 60%, 100% {
            opacity: 0.3 !important;
            transform: translateY(0) !important;
        }
        30% {
            opacity: 1 !important;
            transform: translateY(-8px) !important;
        }
    }

    /* Auto-scroll to bottom functionality */
    .auto-scroll {
        scroll-behavior: smooth !important;
    }

    /* Ensure messages display fully */
    .message-wrapper,
    .message-content,
    .message-text {
        max-height: none !important;
        overflow: visible !important;
    }

    /* Fix any potential height constraints without affecting scrollable areas */
    .stMarkdown div,
    .element-container div:not(.messages-container):not(.chat-box):not(.chat-container) {
        max-height: none !important;
        overflow: visible !important;
        height: auto !important;
    }

    /* Ensure the chat messages area remains scrollable */
    .element-container .messages-container {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        max-height: 100% !important; /* keep it bounded so it scrolls */
    }

    /* Ensure Streamlit doesn't truncate content */
    .stMarkdown {
        max-height: none !important;
        overflow: visible !important;
        height: auto !important;
    }

    /* Force full content display */
    .message-wrapper * {
        max-height: none !important;
        overflow: visible !important;
        height: auto !important;
    }

    /* Override any Streamlit truncation */
    [data-testid="stMarkdownContainer"] {
        max-height: none !important;
        overflow: visible !important;
        height: auto !important;
    }

    /* Custom scrollbar for messages container */
    .messages-container::-webkit-scrollbar {
        width: 8px !important;
    }

    .messages-container::-webkit-scrollbar-track {
        background: #2d2d2d !important;
        border-radius: 4px !important;
    }

    .messages-container::-webkit-scrollbar-thumb {
        background: #565869 !important;
        border-radius: 4px !important;
    }

    .messages-container::-webkit-scrollbar-thumb:hover {
        background: #6b6c7b !important;
    }
</style>

<script>
// Auto-scroll to bottom of messages container
function scrollToBottom() {
    const messagesContainer = document.querySelector('.messages-container');
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// Scroll to bottom when page loads or updates
document.addEventListener('DOMContentLoaded', scrollToBottom);
window.addEventListener('load', scrollToBottom);

// Use MutationObserver to detect when new messages are added
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
            // Check if a message was added
            for (let node of mutation.addedNodes) {
                if (node.nodeType === 1 && (node.classList.contains('message-wrapper') || node.querySelector('.message-wrapper'))) {
                    setTimeout(scrollToBottom, 100);
                    break;
                }
            }
        }
    });
});

// Start observing when the messages container is available
function startObserving() {
    const messagesContainer = document.querySelector('.messages-container');
    if (messagesContainer) {
        observer.observe(messagesContainer, {
            childList: true,
            subtree: true
        });
    } else {
        // Retry after a short delay if container not found
        setTimeout(startObserving, 500);
    }
}

startObserving();
</script>
""", unsafe_allow_html=True)

# Memory and chat management functions
def load_chat_store():
    """Load or create chat store for persistent memory"""
    try:
        if os.path.exists("chat_store.json"):
            return SimpleChatStore.from_persist_path("chat_store.json")
        else:
            return SimpleChatStore()
    except:
        return SimpleChatStore()

def save_chat_store(chat_store):
    """Save chat store to disk"""
    try:
        chat_store.persist("chat_store.json")
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")

def load_chat_sessions():
    """Load chat sessions metadata"""
    try:
        if os.path.exists("chat_sessions.json"):
            with open("chat_sessions.json", "r") as f:
                return json.load(f)
        else:
            return {}
    except:
        return {}

def save_chat_sessions(sessions):
    """Save chat sessions metadata"""
    try:
        with open("chat_sessions.json", "w") as f:
            json.dump(sessions, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save sessions: {e}")

def create_new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": session_id,
        "title": "New Chat",
        "created_at": timestamp,
        "last_updated": timestamp
    }

def update_session_title(sessions, session_id, first_message):
    """Update session title based on first message"""
    if session_id in sessions:
        title = first_message[:30] + "..." if len(first_message) > 30 else first_message
        sessions[session_id]["title"] = title
        sessions[session_id]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def delete_chat_session(session_id):
    """Delete a specific chat session"""
    if session_id in st.session_state.chat_sessions:
        # Remove from sessions
        del st.session_state.chat_sessions[session_id]

        # Clear from chat store
        if hasattr(st.session_state.chat_store, '_store') and session_id in st.session_state.chat_store._store:
            del st.session_state.chat_store._store[session_id]

        # Save updated data
        save_chat_sessions(st.session_state.chat_sessions)
        save_chat_store(st.session_state.chat_store)

        # If we deleted the current session, switch to another one or create new
        if session_id == st.session_state.current_session_id:
            if st.session_state.chat_sessions:
                # Switch to most recent remaining session
                latest_session = max(st.session_state.chat_sessions.items(), key=lambda x: x[1]["last_updated"])
                st.session_state.current_session_id = latest_session[0]

                # Load messages for new current session
                st.session_state.memory = ChatMemoryBuffer.from_defaults(
                    token_limit=40000,
                    chat_store=st.session_state.chat_store,
                    chat_store_key=st.session_state.current_session_id
                )

                try:
                    chat_history = st.session_state.memory.get()
                    if chat_history:
                        st.session_state.messages = [
                            {"role": msg.role, "content": msg.content} for msg in chat_history
                        ]
                    else:
                        st.session_state.messages = []
                except:
                    st.session_state.messages = []
            else:
                # Create a new session if no sessions left
                new_session = create_new_session()
                st.session_state.chat_sessions[new_session["id"]] = new_session
                st.session_state.current_session_id = new_session["id"]

                st.session_state.memory = ChatMemoryBuffer.from_defaults(
                    token_limit=40000,
                    chat_store=st.session_state.chat_store,
                    chat_store_key=new_session["id"]
                )

                st.session_state.messages = []
                save_chat_sessions(st.session_state.chat_sessions)

def get_groq_api_key():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key and hasattr(st, "secrets"):
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
    return api_key

def load_messages_for_session(session_id):
    """Load messages for a specific session"""
    try:
        # Create memory for this session
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=40000,
            chat_store=st.session_state.chat_store,
            chat_store_key=session_id
        )

        # Get chat history from memory
        chat_history = memory.get()
        if chat_history and len(chat_history) > 0:
            return [{"role": msg.role, "content": msg.content} for msg in chat_history]

        # Fallback: try to get directly from chat store
        if hasattr(st.session_state.chat_store, '_store') and session_id in st.session_state.chat_store._store:
            stored_messages = st.session_state.chat_store._store[session_id]
            if stored_messages:
                return [{"role": msg.role, "content": msg.content} for msg in stored_messages]

        return []
    except Exception as e:
        st.sidebar.error(f"Error loading messages: {str(e)}")
        return []

# Initialize session state
if "chat_store" not in st.session_state:
    st.session_state.chat_store = load_chat_store()

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()

if "current_session_id" not in st.session_state:
    # Check if there's an existing empty session to reuse
    empty_session_id = None

    if st.session_state.chat_sessions:
        # Look for an empty session (no messages)
        for session_id, session_data in st.session_state.chat_sessions.items():
            messages = load_messages_for_session(session_id)
            if not messages or len(messages) == 0:
                empty_session_id = session_id
                break

    if empty_session_id:
        # Reuse the empty session
        st.session_state.current_session_id = empty_session_id
    else:
        # Create a new session only if no empty session exists
        new_session = create_new_session()
        st.session_state.chat_sessions[new_session["id"]] = new_session
        st.session_state.current_session_id = new_session["id"]
        save_chat_sessions(st.session_state.chat_sessions)

if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(
        token_limit=40000,
        chat_store=st.session_state.chat_store,
        chat_store_key=st.session_state.current_session_id
    )

# Initialize delete confirmation state
if "delete_confirmation" not in st.session_state:
    st.session_state.delete_confirmation = None

# Initialize messages - start with empty for new session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load query engine and LLM
if "query_engine" not in st.session_state:
    with st.spinner("🕉️ Loading Ayurvedic Knowledge"):
        @st.cache_resource
        def load_embedding_model():
            return FastEmbedEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                embed_batch_size=32,
                cache_dir="./embedding_cache"
            )
        Settings.embed_model = load_embedding_model()

        vector_store = FaissVectorStore.from_persist_dir("faiss_db")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir="faiss_db"
        )

        index = load_index_from_storage(
            storage_context=storage_context,
            index_id="2a3e044a-5744-41d0-9873-8d679b1571a8"
        )

        api_key = get_groq_api_key()
        if not api_key:
            st.error("Please set your GROQ_API_KEY in the .env file or as an environment variable")
            st.stop()

        llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)
        st.session_state.llm = llm

        ayurveda_prompt_str = """
Role: You are an expert Ayurvedic physician and educator.

Core principle:
- Answer only what the user is asking now. Do not continue previous topics unless explicitly asked.
- Be concise, practical, and structured. No greetings or fluff. Use bullet points.

Context (if any): {context_str}
User question: {query_str}

Decision logic:
- If the user asks about a condition/symptom and wants help/remedy (e.g., “I have indigestion; what’s the remedy?”):
  1) Information
  2) Ayurvedic Remedies
  3) Benefits/Expected Outcomes
- If the user asks only about an herb/compound/food (e.g., “What is Ashwagandha?”):
  - Provide a concise profile of that item only (what it is, Ayurvedic properties, primary uses, typical dose, safety).
- If the user asks a general informational question (definition/meaning/differences):
  - Give a direct explanation. Skip remedies/benefits unless specifically requested.
- If the user asks only for a remedy:
  - Give a one-line context if needed, then Remedies and Benefits; keep it focused.
- If essential info is missing and would change advice (e.g., pregnancy, severe illness, age for dosing, current meds):
  - Ask up to 3 concise clarifying questions. If not critical, proceed with safest general guidance.

Formatting rules:
- Use numbered headings when applicable:
  1) Information
  2) Ayurvedic Remedies
  3) Benefits/Expected Outcomes
- Use short bullet points; avoid long paragraphs.
- Include Sanskrit names with common English names and Latin botanical names where relevant.
- Give practical, actionable details:
  - Form, dose range, units, frequency, timing (e.g., before/after meals), and typical duration.
  - Simple preparation/usage steps for teas/decoctions/pastes/externals.
- Always include:
  - Avoid/Precautions (e.g., pregnancy, children, specific conditions)
  - Possible interactions (e.g., anticoagulants, thyroid/BP/diabetes meds)
  - When to seek medical care (red flags)
- Keep language simple and direct.

Section templates:

If CONDITION/SYMPTOM (use all 3 sections):
1) Information
- Ayurvedic perspective (doshas/agni/dhatus involved)
- Common causes (Nidana) and key symptoms
- Red flags: when to seek urgent care

2) Ayurvedic Remedies
- Herbs/formulations (Sanskrit | English | Latin), with dose, form, frequency, duration
- Diet guidelines (pathya/apathya): foods to favor/avoid
- Lifestyle: routines, sleep, mealtime habits
- Yoga/pranayama: 2–5 specific options with brief cues
- External or home therapies (if safe): how to prepare/use
- Panchakarma: only if appropriate and clearly marked “under supervision”
- Avoid/Precautions + Interactions

3) Benefits/Expected Outcomes
- How the remedy supports balance (mechanism in Ayurvedic terms)
- Expected timeline and what improvements to monitor
- Evidence snapshot (e.g., small RCTs/observational/animal; say “limited/mixed” if unclear)

If HERB/COMPOUND/FOOD PROFILE (concise; no full 3-part unless asked for treatment):
- What it is (Sanskrit | English | Latin), part used, forms
- Ayurvedic properties: Rasa, Guna, Virya, Vipaka, Dosha effects
- Primary uses/indications
- Typical adult dose ranges and how to take
- Safety: side effects, who should avoid, interactions
- Evidence snapshot (brief, honest)

If GENERAL EXPLANATION:
- Direct definition/overview in bullets, with key distinctions or examples if useful.

Safety and scope:
- Do not claim cures. Say “may help/support.”
- Do not prescribe metals/mineral bhasma or strong rasayana without supervision.
- Pediatric dosing: provide only if age/weight is given; otherwise say it requires clinician guidance.
- For severe/acute symptoms (e.g., chest pain, severe abdominal pain, high fever, bleeding, neurological deficits), direct the user to urgent care.
- Align with users context if provided (vegetarian, allergies, region).
- If evidence is limited/inconclusive, state that plainly.

Style:
- Start directly with the answer.
- Use the structure appropriate to the question type.
- Correctly interpret common misspellings (e.g., ashwaganda → Ashwagandha).
- Be specific, measurable, and clear.

Now, answer the user question following the above rules and structure.
"""

        ayurveda_prompt = PromptTemplate(ayurveda_prompt_str)

        st.session_state.query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=3,
            text_qa_template=ayurveda_prompt
        )

# Helper functions
def is_greeting_or_chat(query):
    if not query:
        return False

    query_lower = query.lower().strip()

    # Greetings and casual conversation patterns
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                 'namaste', 'how are you', "what's up", 'greetings', 'thank you', 'thanks',
                 'bye', 'goodbye', 'see you', 'ok', 'okay', 'yes', 'no']

    # Personal introductions
    introductions = ['my name is', 'i am', "i'm", 'call me', 'this is']

    # Simple questions about previous conversation
    memory_questions = ['what did i ask', 'what was my', 'do you remember', 'earlier i',
                       'what is my name', 'what\'s my name', 'my name', 'who am i',
                       'what did we discuss', 'what were we talking about', 'previous question',
                       'before this', 'last question', 'what was i asking', 'did i tell you',
                       'did i mention', 'did i say', 'have i told you', 'have i mentioned',
                       'have i asked', 'what question did i ask', 'what did i tell you',
                       'see the previous', 'previous conversations', 'our conversation',
                       'what we discussed']

    # Check for exact matches or starts with greetings
    for greeting in greetings:
        if query_lower == greeting or query_lower.startswith(greeting):
            return True

    # Check for introductions
    for intro in introductions:
        if intro in query_lower:
            return True

    # Check for memory/conversation questions
    for mem_q in memory_questions:
        if mem_q in query_lower:
            return True

    return False

def get_greeting_response(query):
    query_lower = (query or "").lower().strip()

    # Check if user is introducing themselves
    if any(intro in query_lower for intro in ['my name is', 'i am', "i'm", 'call me']):
        # Extract name if possible
        name = ""
        for intro in ['my name is', 'i am', "i'm", 'call me']:
            if intro in query_lower:
                name_part = query_lower.split(intro)[-1].strip()
                if name_part:
                    name = name_part.split()[0].title()  # Get first word and capitalize
                break

        if name:
            return f" Namaste {name}! It's wonderful to meet you. I'm here to help you with Ayurvedic health guidance. How are you feeling today? Any health concerns or wellness topics you'd like to explore?"
        else:
            return " Namaste! Nice to meet you. I'm your Ayurvedic health assistant. How can I help you with your wellness journey today?"

    # Check for memory/conversation questions - expanded list
    elif any(mem_q in query_lower for mem_q in ['what did i ask', 'what was my', 'do you remember', 'earlier i', 'what is my name', 'what\'s my name', 'my name', 'who am i', 'what did we discuss', 'what were we talking about', 'previous question', 'before this', 'last question', 'what was i asking', 'did i tell you', 'did i mention', 'did i say', 'have i told you', 'have i mentioned', 'have i asked', 'what question did i ask', 'what did i tell you', 'see the previous', 'previous conversations', 'our conversation', 'what we discussed']):
        # Handle name questions specifically
        if any(name_q in query_lower for name_q in ['what is my name', 'what\'s my name', 'my name', 'who am i']):
            # Look for name in conversation history
            if hasattr(st.session_state, 'messages') and st.session_state.messages:
                for msg in st.session_state.messages:
                    if msg['role'] == 'user':
                        msg_lower = msg['content'].lower()
                        # Check for name introductions
                        for intro in ['my name is', 'i am', "i'm", 'call me']:
                            if intro in msg_lower:
                                name_part = msg_lower.split(intro)[-1].strip()
                                if name_part:
                                    name = name_part.split()[0].title()
                                    return f"Your name is {name}! I remember when you introduced yourself earlier. How can I help you with your health and wellness today?"

                return "I don't recall you mentioning your name yet in our conversation. Would you like to tell me your name? I'd love to know what to call you!"
            else:
                return "We just started chatting! You haven't told me your name yet. What should I call you?"

        # Handle other memory questions
        else:
            # Look at recent conversation history (EXCLUDING the current question)
            if hasattr(st.session_state, 'messages') and st.session_state.messages:
                # Get all messages except the current one that's being processed
                all_messages = st.session_state.messages[:]  # Copy the list
                user_questions = [msg['content'] for msg in all_messages if msg['role'] == 'user']

                # Check for specific topic questions like "did i tell you about indigestion"
                if any(topic_q in query_lower for topic_q in ['did i tell you', 'did i mention', 'did i say', 'have i told you', 'have i mentioned']):
                    # Extract the topic they're asking about
                    topic_keywords = []
                    for keyword in ['indigestion', 'digestion', 'stomach', 'health', 'problem', 'issue', 'pain', 'ache', 'diet', 'food', 'medicine', 'remedy', 'treatment', 'ayurveda', 'dosha', 'pitta', 'vata', 'kapha']:
                        if keyword in query_lower:
                            topic_keywords.append(keyword)

                    # Search through conversation for mentions of these topics
                    found_mentions = []
                    for i, msg in enumerate(all_messages):
                        if msg['role'] == 'user':
                            msg_lower = msg['content'].lower()
                            for keyword in topic_keywords:
                                if keyword in msg_lower:
                                    found_mentions.append((i, msg['content'], keyword))

                    if found_mentions:
                        # Show the relevant previous messages
                        if len(found_mentions) == 1:
                            return f"Yes, you mentioned that earlier! You said: \"{found_mentions[0][1]}\"\n\nWould you like me to provide more information about this topic?"
                        else:
                            mentions_text = "\n".join([f"• {mention[1]}" for mention in found_mentions])
                            return f"Yes, you've mentioned this topic before in our conversation:\n\n{mentions_text}\n\nWould you like me to elaborate on any of these points?"
                    else:
                        if topic_keywords:
                            return f"I don't see any previous mention of {', '.join(topic_keywords)} in our current conversation. Would you like to tell me about it now?"
                        else:
                            return "I don't see any previous mention of that topic in our current conversation. What would you like to discuss?"

                if len(user_questions) >= 1:
                    # Show the most recent previous question(s)
                    if len(user_questions) == 1:
                        return f"In our conversation, you asked: \"{user_questions[0]}\". Would you like me to elaborate on that topic or do you have a new question?"
                    else:
                        # Show last 2-3 questions for better context
                        recent_questions = user_questions[-3:] if len(user_questions) >= 3 else user_questions
                        questions_text = "\n".join([f"• {q}" for q in recent_questions])
                        return f"Here's what you've asked in our conversation:\n\n{questions_text}\n\nWhich topic would you like me to elaborate on, or do you have a new question?"
                else:
                    return "We just started our conversation! You haven't asked any health questions yet. What would you like to know about Ayurveda or your wellness?"
            else:
                return "This is the beginning of our conversation. What Ayurvedic topic or health concern would you like to discuss?"

    # Regular greetings
    elif any(x in query_lower for x in ['hello', 'hi', 'hey']):
        return " Namaste! How can I assist you with your health and wellness today? Feel free to ask me about Ayurvedic remedies, dietary advice, or any health concerns."
    elif 'how are you' in query_lower:
        return " I'm here and ready to help you on your wellness journey! How are you feeling today? Any health concern or Ayurvedic topic you'd like to explore?"
    elif 'good morning' in query_lower:
        return "🌅 Good morning! Would you like some Ayurvedic morning routine (Dinacharya) tips or help with any health questions?"
    elif 'good afternoon' in query_lower or 'good evening' in query_lower:
        return " Namaste! I hope you're having a peaceful day. How may I assist you with your health and wellness needs?"
    elif 'thank' in query_lower:
        return " You're most welcome! Is there anything else about Ayurveda or your health that you'd like to know?"
    elif 'bye' in query_lower or 'goodbye' in query_lower:
        return " Namaste! Take care of your health and well-being. Feel free to return anytime you need Ayurvedic guidance."
    else:
        return " I'm here to help! Ask me about Ayurvedic medicine, natural remedies, diet, or wellness practices."

def is_follow_up_question(query):
    if not query:
        return False

    query_lower = query.lower().strip()

    # Direct follow-up phrases
    follow_up_words = ['elaborate', 'more about this', 'explain more', 'tell me more',
                       'what else', 'continue', 'go on', 'expand on that', 'more details',
                       'can you explain', 'what about', 'how about', 'also tell me',
                       'anything else', 'more information']

    # Question words that often indicate follow-ups
    follow_up_questions = ['what other', 'what are some', 'are there any', 'can you suggest',
                          'do you have', 'what would you recommend', 'how can i', 'should i']

    # Check for direct follow-up phrases
    for phrase in follow_up_words:
        if phrase in query_lower:
            return True

    # Check for follow-up question patterns
    for question in follow_up_questions:
        if query_lower.startswith(question):
            return True

    # Check if it's a short question (likely a follow-up)
    if len(query.split()) <= 3 and any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
        return True

    return False

# Simple delete confirmation in sidebar
if st.session_state.delete_confirmation:
    with st.sidebar:
        st.markdown("---")
        st.warning(f"Delete '{st.session_state.chat_sessions[st.session_state.delete_confirmation]['title']}'?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", key="cancel_delete"):
                st.session_state.delete_confirmation = None
                st.rerun()

        with col2:
            if st.button("Delete", key="confirm_delete", type="primary"):
                session_to_delete = st.session_state.delete_confirmation
                st.session_state.delete_confirmation = None
                delete_chat_session(session_to_delete)
                st.rerun()

# Sidebar for chat history
with st.sidebar:
    # New chat button
    if st.button("+ New Chat", key="new_chat", help="Start a new conversation"):
        new_session = create_new_session()
        st.session_state.chat_sessions[new_session["id"]] = new_session
        st.session_state.current_session_id = new_session["id"]

        st.session_state.memory = ChatMemoryBuffer.from_defaults(
            token_limit=40000,
            chat_store=st.session_state.chat_store,
            chat_store_key=new_session["id"]
        )

        st.session_state.messages = []
        save_chat_sessions(st.session_state.chat_sessions)
        save_chat_store(st.session_state.chat_store)
        st.rerun()

    # Chat history
    st.markdown('<div style="color: #8e8ea0; font-size: 12px; font-weight: 600; margin: 16px 0 8px 0; text-transform: uppercase; letter-spacing: 0.5px;">Chat History</div>', unsafe_allow_html=True)

    if st.session_state.chat_sessions:
        sorted_sessions = sorted(
            st.session_state.chat_sessions.items(),
            key=lambda x: x[1]["last_updated"],
            reverse=True
        )

        for session_id, session_data in sorted_sessions:
            is_current = session_id == st.session_state.current_session_id

            # Only show delete confirmation for one session at a time
            if st.session_state.delete_confirmation != session_id:
                # Create columns for chat item and delete button
                col1, col2 = st.columns([4, 1])

                with col1:
                    if st.button(
                        session_data['title'],
                        key=f"session_{session_id}",
                        help=f"Switch to: {session_data['title']}",
                        use_container_width=True
                    ):
                        # Switch to the selected session
                        st.session_state.current_session_id = session_id

                        # Create new memory instance for this session
                        st.session_state.memory = ChatMemoryBuffer.from_defaults(
                            token_limit=40000,
                            chat_store=st.session_state.chat_store,
                            chat_store_key=session_id
                        )

                        # Load messages for this session
                        st.session_state.messages = load_messages_for_session(session_id)

                        st.rerun()

                with col2:
                    # Only show delete button if there's more than one session
                    if len(st.session_state.chat_sessions) > 1:
                        if st.button("🗑️", key=f"delete_{session_id}", help="Delete Chat"):
                            st.session_state.delete_confirmation = session_id
                            st.rerun()

    # Add clear all chats option at bottom
    if len(st.session_state.chat_sessions) > 1:
        st.markdown("---")
        if st.button("🗑️ Clear All Chats", help="Delete all chat history", use_container_width=True):
            # Keep only current session, delete all others
            current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
            st.session_state.chat_sessions = {st.session_state.current_session_id: current_session}

            # Clear chat store for all sessions except current
            if hasattr(st.session_state.chat_store, '_store'):
                st.session_state.chat_store._store = {st.session_state.current_session_id: st.session_state.chat_store._store.get(st.session_state.current_session_id, [])}

            # Clear current session messages too
            st.session_state.messages = []
            st.session_state.memory = ChatMemoryBuffer.from_defaults(
                token_limit=40000,
                chat_store=st.session_state.chat_store,
                chat_store_key=st.session_state.current_session_id
            )

            save_chat_sessions(st.session_state.chat_sessions)
            save_chat_store(st.session_state.chat_store)
            st.rerun()

    # Debug information (remove this later)
    with st.expander("Debug Info", expanded=False):
        st.write(f"Current Session: {st.session_state.current_session_id}")
        st.write(f"Messages Count: {len(st.session_state.messages) if st.session_state.messages else 0}")
        if st.session_state.messages:
            st.write("All messages in current session:")
            for i, msg in enumerate(st.session_state.messages):
                st.write(f"{i+1}: {msg['role']}: {msg['content']}")

        # Show chat store contents
        if hasattr(st.session_state.chat_store, '_store'):
            st.write(f"Chat Store Sessions: {list(st.session_state.chat_store._store.keys())}")
            if st.session_state.current_session_id in st.session_state.chat_store._store:
                stored_count = len(st.session_state.chat_store._store[st.session_state.current_session_id])
                st.write(f"Stored messages for current session: {stored_count}")

        # Memory test
        if st.button("🧠 Test Memory", key="test_memory"):
            if st.session_state.messages:
                user_msgs = [msg['content'] for msg in st.session_state.messages if msg['role'] == 'user']
                st.write(f"User questions found: {len(user_msgs)}")
                for i, q in enumerate(user_msgs):
                    st.write(f"{i+1}: {q}")
            else:
                st.write("No messages in memory")

        # Force refresh button
        if st.button("🔄 Force Refresh Messages", key="force_refresh"):
            st.session_state.messages = load_messages_for_session(st.session_state.current_session_id)
            st.rerun()

# Main chat interface (always show, delete confirmation is in sidebar)
if True:  # Always show main interface
    # Create main chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Create chat box container
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)

    # Messages container
    # Build messages HTML in a single block so all items are true children of the container
    messages_html_parts = ['<div class="messages-container auto-scroll">']

    if st.session_state.messages:
        for message in st.session_state.messages:
            if message.get("role") == "user":
                user_content = html.escape(message.get("content", ""))
                messages_html_parts.append(textwrap.dedent(f"""
<div class=\"message-wrapper user\">
  <div class=\"message-avatar user-avatar\">👤</div>
  <div class=\"message-text\">{user_content}</div>
</div>
"""))
            else:
                content = (message.get("content") or "").strip()
                if not content:
                    content = "I apologize, but I couldn't generate a response. Please try asking your question again."
                assistant_content = html.escape(content).replace('\n', '<br>')
                messages_html_parts.append(textwrap.dedent(f"""
<div class=\"message-wrapper assistant\">
  <div class=\"message-avatar assistant-avatar\">🕉️</div>
  <div class=\"message-text\">{assistant_content}</div>
</div>
"""))
    else:
        messages_html_parts.append(textwrap.dedent("""
<div class=\"welcome-container\">
  <div class=\"welcome-title\"> Ayurvedic Health Assistant</div>
  <div class=\"welcome-subtitle\">
    Welcome to your personal Ayurvedic health companion. Ask me about natural remedies,
    dietary advice, lifestyle recommendations, or any health concerns from an Ayurvedic perspective.
  </div>
</div>
"""))

    messages_html_parts.append('</div>')  # close messages-container
    st.markdown("\n".join(messages_html_parts), unsafe_allow_html=True)

    # Chat input area
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask about Ayurvedic remedies, diet, or health concerns..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.memory.put(ChatMessage(role="user", content=prompt))

        # Save immediately to ensure persistence
        save_chat_store(st.session_state.chat_store)

        # Update session title if this is the first message
        if len(st.session_state.messages) == 1:
            update_session_title(st.session_state.chat_sessions, st.session_state.current_session_id, prompt)
            save_chat_sessions(st.session_state.chat_sessions)

        # Generate response
        with st.spinner(" Consulting Ayurvedic wisdom"):
            try:
                if is_greeting_or_chat(prompt):
                    response = get_greeting_response(prompt)
                elif is_follow_up_question(prompt) and len(st.session_state.messages) > 1:
                    # For follow-up questions, include recent context
                    recent_context = ""
                    if len(st.session_state.messages) >= 3:
                        # Get last few exchanges for context
                        last_user_msg = st.session_state.messages[-3]['content'] if st.session_state.messages[-3]['role'] == 'user' else ""
                        last_assistant_msg = st.session_state.messages[-2]['content'] if st.session_state.messages[-2]['role'] == 'assistant' else ""

                        recent_context = f"Previous conversation context:\nUser asked: {last_user_msg}\nI responded: {last_assistant_msg}\n\nNow the user is asking: {prompt}\n\nPlease provide a helpful follow-up response that builds on our previous discussion."
                    else:
                        recent_context = f"This is a follow-up to our conversation. User is now asking: {prompt}"

                    response = str(st.session_state.query_engine.query(recent_context))
                else:
                    # For regular health questions, provide context about this being part of a conversation
                    if len(st.session_state.messages) > 1:
                        conversation_context = f"We are having a conversation about health and Ayurveda. The user is now asking: {prompt}\n\nPlease provide a helpful, conversational response."
                        response = str(st.session_state.query_engine.query(conversation_context))
                    else:
                        response = str(st.session_state.query_engine.query(prompt))

                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.memory.put(ChatMessage(role="assistant", content=response))

                # Save chat store
                save_chat_store(st.session_state.chat_store)

            except Exception as e:
                error_msg = f"I apologize, but I encountered an error while processing your question. Please try again. Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-input-wrapper
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-input-container
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-box
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container
