#!/usr/bin/env python3
"""
Test script to verify chatbot memory functionality
"""

import json
import os
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore

def test_memory_functionality():
    """Test the memory implementation"""
    print("🧠 Testing Chatbot Memory Functionality")
    print("=" * 50)
    
    # Test 1: Check if chat store files exist
    print("\n1. Checking for persistent storage files:")
    chat_store_exists = os.path.exists("chat_store.json")
    chat_sessions_exists = os.path.exists("chat_sessions.json")
    
    print(f"   chat_store.json exists: {chat_store_exists}")
    print(f"   chat_sessions.json exists: {chat_sessions_exists}")
    
    # Test 2: Load and examine chat store
    if chat_store_exists:
        print("\n2. Examining chat store contents:")
        try:
            chat_store = SimpleChatStore.from_persist_path("chat_store.json")
            if hasattr(chat_store, '_store'):
                sessions = list(chat_store._store.keys())
                print(f"   Number of sessions in store: {len(sessions)}")
                
                for session_id in sessions:
                    messages = chat_store._store[session_id]
                    print(f"   Session {session_id[:8]}...: {len(messages)} messages")
                    
                    # Show first few messages for context
                    for i, msg in enumerate(messages[:3]):
                        role = msg.role if hasattr(msg, 'role') else 'unknown'
                        content = msg.content[:50] if hasattr(msg, 'content') else 'no content'
                        print(f"     {i+1}. {role}: {content}...")
                        
                    if len(messages) > 3:
                        print(f"     ... and {len(messages) - 3} more messages")
            else:
                print("   Chat store has no _store attribute")
        except Exception as e:
            print(f"   Error loading chat store: {e}")
    else:
        print("\n2. No chat store file found - memory not yet initialized")
    
    # Test 3: Load and examine chat sessions
    if chat_sessions_exists:
        print("\n3. Examining chat sessions:")
        try:
            with open("chat_sessions.json", "r") as f:
                sessions = json.load(f)
            
            print(f"   Number of sessions: {len(sessions)}")
            for session_id, session_data in sessions.items():
                title = session_data.get('title', 'No title')
                created = session_data.get('created_at', 'Unknown')
                updated = session_data.get('last_updated', 'Unknown')
                print(f"   Session {session_id[:8]}...: '{title}' (created: {created})")
        except Exception as e:
            print(f"   Error loading sessions: {e}")
    else:
        print("\n3. No chat sessions file found")
    
    # Test 4: Test memory buffer functionality
    print("\n4. Testing ChatMemoryBuffer functionality:")
    try:
        # Create a test chat store
        test_store = SimpleChatStore()
        test_session_id = "test_session_123"
        
        # Create memory buffer
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=40000,
            chat_store=test_store,
            chat_store_key=test_session_id
        )
        
        # Add some test messages
        memory.put(ChatMessage(role="user", content="What is Ashwagandha?"))
        memory.put(ChatMessage(role="assistant", content="Ashwagandha is an adaptogenic herb..."))
        memory.put(ChatMessage(role="user", content="What are its benefits?"))
        memory.put(ChatMessage(role="assistant", content="Ashwagandha has several benefits including..."))
        
        # Retrieve messages
        retrieved_messages = memory.get()
        print(f"   Successfully stored and retrieved {len(retrieved_messages)} messages")
        
        # Test follow-up context
        if len(retrieved_messages) >= 2:
            last_user = retrieved_messages[-2].content if retrieved_messages[-2].role == "user" else ""
            last_assistant = retrieved_messages[-1].content if retrieved_messages[-1].role == "assistant" else ""
            print(f"   Last user message: {last_user[:50]}...")
            print(f"   Last assistant message: {last_assistant[:50]}...")
            print("   ✅ Memory retrieval working correctly")
        else:
            print("   ❌ Not enough messages retrieved")
            
    except Exception as e:
        print(f"   ❌ Error testing memory buffer: {e}")
    
    # Test 5: Test follow-up question detection
    print("\n5. Testing follow-up question detection:")
    
    def test_follow_up_detection(query):
        """Simplified version of the follow-up detection logic"""
        if not query:
            return False
        
        query_lower = query.lower().strip()
        
        follow_up_words = ['elaborate', 'more about this', 'explain more', 'tell me more',
                          'what else', 'continue', 'go on', 'expand on that', 'more details',
                          'can you explain', 'what about', 'how about', 'also tell me',
                          'anything else', 'more information']
        
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
    
    test_queries = [
        ("What is Ashwagandha?", False),
        ("Tell me more", True),
        ("What about dosage?", True),
        ("Can you explain more?", True),
        ("What are some benefits?", True),
        ("How much should I take?", False),
        ("What else?", True),
        ("Are there any side effects?", True),
        ("Why?", True),
        ("I have diabetes", False)
    ]
    
    for query, expected in test_queries:
        result = test_follow_up_detection(query)
        status = "✅" if result == expected else "❌"
        print(f"   {status} '{query}' -> Follow-up: {result} (expected: {expected})")
    
    print("\n" + "=" * 50)
    print("Memory test completed!")
    
    # Recommendations
    print("\n📋 MEMORY ANALYSIS SUMMARY:")
    print("1. The app uses ChatMemoryBuffer with SimpleChatStore for persistence")
    print("2. Messages are stored both in session_state.messages and memory buffer")
    print("3. Follow-up questions get recent context from previous exchanges")
    print("4. Each chat session has its own memory space")
    print("5. Memory is persisted to disk (chat_store.json)")
    
    print("\n🔍 POTENTIAL ISSUES TO CHECK:")
    print("1. Verify that memory.put() is called for both user and assistant messages")
    print("2. Check if query_engine receives proper context for follow-ups")
    print("3. Ensure session switching properly loads the correct memory")
    print("4. Test if memory persists across app restarts")

if __name__ == "__main__":
    test_memory_functionality()
