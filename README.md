# Ayurvedic Health Assistant Chatbot

A WhatsApp-style chatbot interface built with Streamlit that provides Ayurvedic health advice and natural remedies based on ancient wisdom.

## Features

- 🌿 WhatsApp-style chat interface
- 🧠 AI-powered responses using Groq's Llama 3.3 70B model
- 📚 Knowledge base powered by FAISS vector store
- 🔍 FastEmbed embeddings for semantic search
- 🎨 Beautiful, responsive UI with custom CSS
- 💬 Persistent chat history with session management
- 🗑️ Delete individual chats or clear all history

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ayurvedic-health-assistant.git
   cd ayurvedic-health-assistant
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
ayurvedic-health-assistant/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .env                  # Environment variables (not in repo)
├── .gitignore           # Git ignore file
├── faiss_db/            # FAISS vector database
│   ├── default__vector_store.json
│   ├── docstore.json
│   ├── graph_store.json
│   ├── image__vector_store.json
│   └── index_store.json
├── embedding_cache/     # Cached embedding models
│   └── models--qdrant--bge-small-en-v1.5-onnx-q/
├── chat_store.json      # Persistent chat history
└── chat_sessions.json   # Chat session metadata
```

## Dependencies

- **streamlit**: Web interface framework
- **llama-index**: LLM application framework
- **llama-index-embeddings-fastembed**: Fast embedding models
- **llama-index-vector-stores-faiss**: FAISS vector store integration
- **llama-index-llms-groq**: Groq LLM integration
- **faiss-cpu**: Vector similarity search
- **python-dotenv**: Environment variable management

## Usage

1. Start the application using `streamlit run app.py`
2. Open your browser to the provided URL (usually `http://localhost:8501`)
3. Start chatting with the Ayurvedic Health Assistant
4. Ask questions about:
   - Natural remedies
   - Dietary advice
   - Ayurvedic treatments
   - Health conditions
   - Wellness practices

## Security

- API keys are loaded from environment variables
- Sensitive files are excluded via `.gitignore`
- No hardcoded credentials in the source code

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and demonstration purposes.
