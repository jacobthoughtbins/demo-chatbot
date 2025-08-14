#!/usr/bin/env python3
"""
Test the improved follow-up question detection
"""

def is_follow_up_question(query):
    """Improved follow-up detection logic"""
    if not query:
        return False

    query_lower = query.lower().strip()

    # Direct follow-up phrases
    follow_up_words = ['elaborate', 'more about this', 'explain more', 'tell me more',
                       'what else', 'continue', 'go on', 'expand on that', 'more details',
                       'can you explain', 'what about', 'how about', 'also tell me',
                       'anything else', 'more information', 'you mentioned', 'you said',
                       'you told me', 'you recommended', 'you suggested', 'from before',
                       'earlier you', 'previously you']

    # Question words that often indicate follow-ups
    follow_up_questions = ['what other', 'what are some', 'are there any', 'can you suggest',
                          'do you have', 'what would you recommend', 'how can i', 'should i',
                          'what herbs', 'what remedies', 'what medicines', 'what treatments',
                          'which herbs', 'which remedies', 'what did you', 'which did you']

    # Check for direct follow-up phrases
    for phrase in follow_up_words:
        if phrase in query_lower:
            return True

    # Check for follow-up question patterns
    for question in follow_up_questions:
        if query_lower.startswith(question):
            return True

    # Check for reference questions (asking about previous responses)
    reference_patterns = ['what herbs did you', 'what remedies did you', 'what medicines did you',
                         'which herbs did you', 'which remedies did you', 'what did you mention',
                         'what did you recommend', 'what did you suggest', 'what did you say']
    
    for pattern in reference_patterns:
        if pattern in query_lower:
            return True

    # Check if it's a short question (likely a follow-up)
    if len(query.split()) <= 3 and any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
        return True

    return False

def test_follow_up_detection():
    print("🧪 TESTING IMPROVED FOLLOW-UP DETECTION")
    print("=" * 50)
    
    test_cases = [
        # The problematic case from your example
        ("what herbs did you mentioned ?", True, "❌ FAILED BEFORE"),
        ("what herbs did you mention?", True, "Should work now"),
        ("what remedies did you suggest?", True, "Should work now"),
        ("what did you recommend?", True, "Should work now"),
        
        # Other follow-up patterns
        ("tell me more", True, "Should work"),
        ("what about dosage?", True, "Should work"),
        ("you mentioned turmeric", True, "Should work"),
        ("what else?", True, "Should work"),
        
        # Non-follow-up questions
        ("what is diabetes?", False, "Not a follow-up"),
        ("hello", False, "Not a follow-up"),
        ("my name is john", False, "Not a follow-up"),
    ]
    
    print("\n📋 TEST RESULTS:")
    print("-" * 30)
    
    for query, expected, note in test_cases:
        result = is_follow_up_question(query)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{query}' -> {result} (expected: {expected}) | {note}")
    
    print(f"\n🎯 SPECIFIC FIX FOR YOUR ISSUE:")
    print("-" * 35)
    
    problematic_query = "what herbs did you mentioned ?"
    result = is_follow_up_question(problematic_query)
    print(f"Query: '{problematic_query}'")
    print(f"Detected as follow-up: {result}")
    print(f"Status: {'✅ FIXED!' if result else '❌ Still broken'}")
    
    print(f"\n📝 WHAT THIS MEANS:")
    print("- The bot will now detect 'what herbs did you mention?' as a follow-up")
    print("- It will include recent conversation context")
    print("- It should reference the diabetes herbs (Turmeric, Gymnema, Fenugreek)")
    print("- Instead of giving random herbs from the knowledge base")

if __name__ == "__main__":
    test_follow_up_detection()
