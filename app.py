import streamlit as st
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.core.prompts import PromptTemplate
import faiss
import os

# Page configuration
st.set_page_config(
    page_title="Ayurvedic Health Assistant",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for WhatsApp-style UI with BLACK text
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .stApp {
        background-color: #e5ddd5;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23d4d4d4' fill-opacity='0.2'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }
    
    /* Chat header */
    .chat-header {
        background: linear-gradient(135deg, #075e54 0%, #128c7e 100%);
        color: white;
        padding: 20px;
        border-radius: 10px 10px 0 0;
        display: flex;
        align-items: center;
        gap: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .header-avatar {
        width: 50px;
        height: 50px;
        background: #25d366;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
    }
    
    .header-info h2 {
        margin: 0;
        font-size: 20px;
        font-weight: 600;
        color: white !important;
    }
    
    .header-info p {
        margin: 0;
        font-size: 14px;
        opacity: 0.9;
        color: #e8f5e9 !important;
    }
    
    /* Force ALL text to be black in chat messages */
    .stChatMessage * {
        color: #000000 !important;
    }
    
    .stChatMessage p,
    .stChatMessage div,
    .stChatMessage span,
    .stChatMessage li,
    .stChatMessage ol,
    .stChatMessage ul {
        color: #000000 !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
    }
    
    /* User messages */
    div[data-testid="user-message"] {
        background-color: #dcf8c6 !important;
        border-radius: 15px 15px 0 15px !important;
        margin-left: 20% !important;
        margin-bottom: 15px !important;
        padding: 12px 16px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    /* Assistant messages */
    div[data-testid="assistant-message"] {
        background-color: #ffffff !important;
        border-radius: 15px 15px 15px 0 !important;
        margin-right: 20% !important;
        margin-bottom: 15px !important;
        padding: 12px 16px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    /* Avatar styling */
    .stChatMessage img {
        width: 35px !important;
        height: 35px !important;
        margin-top: 5px !important;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        background-color: #f0f0f0;
        border-radius: 30px;
        padding: 8px;
        margin: 20px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stChatInputContainer > div {
        background-color: white !important;
        border-radius: 25px !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    .stChatInputContainer textarea {
        color: #000000 !important;
        font-size: 15px !important;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px;
        color: #667781;
        font-size: 14px;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    
    .typing-dots span {
        width: 8px;
        height: 8px;
        background-color: #667781;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            opacity: 0.3;
            transform: translateY(0);
        }
        30% {
            opacity: 1;
            transform: translateY(-10px);
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "🙏 Namaste! I'm your Ayurvedic Health Assistant. I can help you with natural remedies, dietary advice, and holistic wellness tips based on ancient Ayurvedic wisdom. How may I assist you today?"
    })

if "query_engine" not in st.session_state:
    with st.spinner("🌿 Loading Ayurvedic Knowledge Base..."):
        # Cache the embedding model
        @st.cache_resource
        def load_embedding_model():
            return FastEmbedEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                embed_batch_size=32,
                cache_dir="./embedding_cache"
            )
        
        Settings.embed_model = load_embedding_model()
        
        # Load FAISS index
        d = 384
        faiss_index = faiss.IndexFlatL2(d)
        
        # Load vector store
        vector_store = FaissVectorStore.from_persist_dir("faiss_db")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, 
            persist_dir="faiss_db"
        )
        
        # Load index
        index = load_index_from_storage(
            storage_context=storage_context,
            index_id="2a3e044a-5744-41d0-9873-8d679b1571a8"
        )
        
        # Initialize LLM
        llm = Groq(model="llama-3.1-8b-instant", api_key=os.environ.get("Groq_api_key"))
        
        # Store LLM separately for direct chat
        st.session_state.llm = llm
        
        # Improved Ayurvedic prompt
        ayurveda_prompt_str = """You are an expert Ayurvedic physician. 

CRITICAL: Focus ONLY on what the user is currently asking about. If they ask about Apigenin, discuss Apigenin. If they ask about diabetes, discuss diabetes. DO NOT continue previous topics unless explicitly asked.

Context from knowledge base: {context_str}

Current Question: {query_str}

Provide a response following this structure when applicable:

**1. Definition/Information**
- What is it from an Ayurvedic perspective
- Key properties and characteristics
- Causes : what are the causes for the diseases
- Symptoms : if it's a condition

**2. Ayurvedic Treatment/Remedies** (if applicable)
- Herbal medicines with specific dosages
- Dietary recommendations
- Lifestyle modifications
- Yoga/pranayama practices
- Panchakarma therapies

**3. Benefits/Expected Outcomes**
- How it works in the body
- Expected results and timeline
- Scientific backing if available

Guidelines:
- Answer ONLY about the specific topic asked
- Be direct - no greetings or introductions
- Use bullet points for clarity
- Include Sanskrit names with translations
- Provide practical, actionable advice
- If asked about a compound/herb, focus on its properties and uses
- If the topic doesn't fit the structure, adapt accordingly
- End with healthcare consultation reminder for medical conditions

Remember: Each question is independent. Do not reference or continue from previous answers unless the user explicitly asks for elaboration."""
        
        ayurveda_prompt = PromptTemplate(ayurveda_prompt_str)
        
        # Create simple query engine WITHOUT chat memory
        st.session_state.query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=3,
            text_qa_template=ayurveda_prompt
        )

# Function to check if query is a greeting or simple chat
def is_greeting_or_chat(query):
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 
                 'namaste', 'how are you', "what's up", 'greetings', 'thank you', 'thanks',
                 'bye', 'goodbye', 'see you', 'ok', 'okay', 'yes', 'no']
    
    query_lower = query.lower().strip()
    
    # Check for exact matches or if query starts with greeting
    for greeting in greetings:
        if query_lower == greeting or query_lower.startswith(greeting):
            return True
    
    return False

# Function to generate greeting response
def get_greeting_response(query):
    query_lower = query.lower().strip()
    
    if 'hello' in query_lower or 'hi' in query_lower or 'hey' in query_lower:
        return "🙏 Namaste! How can I assist you with your health and wellness today? Feel free to ask me about Ayurvedic remedies, dietary advice, or any health concerns you may have."
    
    elif 'how are you' in query_lower:
        return "🌿 I'm here and ready to help you with your wellness journey! How are you feeling today? Is there any health concern or Ayurvedic topic you'd like to explore?"
    
    elif 'good morning' in query_lower:
        return "🌅 Good morning! Starting the day with awareness is wonderful. Would you like some Ayurvedic morning routine (Dinacharya) tips or help with any health questions?"
    
    elif 'good afternoon' in query_lower or 'good evening' in query_lower:
        return "🙏 Namaste! I hope you're having a peaceful day. How may I assist you with your health and wellness needs?"
    
    elif 'thank' in query_lower:
        return "🙏 You're most welcome! It's my pleasure to help. Is there anything else about Ayurveda or your health that you'd like to know?"
    
    elif 'bye' in query_lower or 'goodbye' in query_lower:
        return "🙏 Namaste! Take care of your health and well-being. Feel free to return anytime you need Ayurvedic guidance. Stay healthy!"
    
    else:
        return "🌿 I'm here to help! Please feel free to ask me any questions about Ayurvedic medicine, natural remedies, diet, or wellness practices."

# Function to check if query needs context from previous conversation
def is_follow_up_question(query):
    """Check if query is asking for elaboration on previous topic"""
    follow_up_words = ['elaborate', 'more about this', 'explain more', 'tell me more', 
                       'what else', 'continue', 'go on', 'expand on that']
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in follow_up_words)

# Chat header
st.markdown("""
<div class="chat-header">
    <div class="header-avatar">🌿</div>
    <div class="header-info">
        <h2>Ayurvedic Health Assistant</h2>
        <p>Online • Ancient wisdom for modern wellness</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Chat messages display
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🌿"):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your health question..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)
    
    # Show typing indicator
    with st.chat_message("assistant", avatar="🌿"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="typing-indicator">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <span>Vaidya is typing...</span>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Check if it's a greeting or simple chat
            if is_greeting_or_chat(prompt):
                # Use direct response for greetings
                assistant_response = get_greeting_response(prompt)
            elif is_follow_up_question(prompt) and len(st.session_state.messages) > 2:
                # For follow-up questions, add context from previous exchange
                prev_question = ""
                for i in range(len(st.session_state.messages) - 2, -1, -1):
                    if st.session_state.messages[i]["role"] == "user":
                        prev_question = st.session_state.messages[i]["content"]
                        break
                
                # Combine previous question with current request
                enhanced_query = f"Previous question was: {prev_question}. Now the user asks: {prompt}"
                response = st.session_state.query_engine.query(enhanced_query)
                assistant_response = str(response)
            else:
                # Use simple query engine for health-related queries
                response = st.session_state.query_engine.query(prompt)
                assistant_response = str(response)
            
            # Clear typing indicator and show response
            typing_placeholder.empty()
            st.write(assistant_response)
            
            # Add to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response
            })
            
        except Exception as e:
            typing_placeholder.empty()
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
