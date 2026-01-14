import streamlit as st
import json
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="SmartBank Support Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load intents from JSON file
@st.cache_resource
def load_intents():
    """Load and cache the intents data"""
    try:
        with open('intents.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data['intents']
    except FileNotFoundError:
        st.error("❌ Error: 'intents.json' file not found! Please ensure it's in the same directory as app.py")
        st.stop()
    except json.JSONDecodeError:
        st.error("❌ Error: 'intents.json' is not valid JSON. Please check the file format.")
        st.stop()

# Text preprocessing function
def preprocess_text(text):
    """Clean and normalize text input"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Initialize TF-IDF Vectorizer and fit on patterns
@st.cache_resource
def initialize_nlp_model(_intents):
    """Initialize and train the TF-IDF model on intent patterns"""
    # Prepare training data
    patterns = []
    pattern_to_intent = []
    
    for intent in _intents:
        for pattern in intent['patterns']:
            patterns.append(preprocess_text(pattern))
            pattern_to_intent.append(intent)
    
    # Initialize and fit vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Use unigrams and bigrams
        max_features=500,
        min_df=1
    )
    
    pattern_vectors = vectorizer.fit_transform(patterns)
    
    return vectorizer, pattern_vectors, pattern_to_intent

# Get chatbot response using NLP
def get_bot_response(user_input, vectorizer, pattern_vectors, pattern_to_intent, confidence_threshold=0.3):
    """
    Process user input and return appropriate response using cosine similarity
    """
    # Preprocess user input
    processed_input = preprocess_text(user_input)
    
    # Vectorize user input
    user_vector = vectorizer.transform([processed_input])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, pattern_vectors).flatten()
    
    # Get best match
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    # If confidence is too low, return fallback message
    if best_similarity < confidence_threshold:
        fallback_responses = [
            "I'm not quite sure I understand. Could you rephrase that?",
            "I didn't catch that. Can you try asking in a different way?",
            "I'm still learning! Could you please clarify your question?",
            "Hmm, I'm not sure about that. Try asking about account balance, transfers, loans, or card issues.",
            "I apologize, but I don't have information on that. Please contact our customer service at 1-800-SMART-BANK for assistance."
        ]
        return random.choice(fallback_responses), best_similarity, "unknown"
    
    # Get the matched intent and return a random response
    matched_intent = pattern_to_intent[best_match_idx]
    response = random.choice(matched_intent['responses'])
    
    return response, best_similarity, matched_intent['tag']

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm SmartBank Support Bot. I'm here to help you with:\n\n✅ Account balance & transactions\n✅ Password reset & login issues\n✅ Money transfers & bill payments\n✅ Loan information\n✅ Card management\n✅ ATM locations\n\nHow can I assist you today?"
    })

# Sidebar
with st.sidebar:
    st.title("🤖 SmartBank Support Bot")
    st.markdown("---")
    
    st.subheader("📊 About This Bot")
    st.write("""
    This intelligent chatbot uses **Natural Language Processing (NLP)** to understand and respond to your banking queries.
    
    **🧠 Technology Stack:**
    - TF-IDF Vectorization
    - Cosine Similarity Matching
    - Scikit-learn ML Library
    - Streamlit UI Framework
    """)
    
    st.markdown("---")
    
    st.subheader("💡 Try asking:")
    st.write("""
    - "What's my account balance?"
    - "I forgot my password"
    - "How do I transfer money?"
    - "I need a loan"
    - "My card is blocked"
    """)
    
    st.markdown("---")
    
    st.subheader("📈 Model Stats")
    intents = load_intents()
    total_patterns = sum(len(intent['patterns']) for intent in intents)
    st.metric("Total Intents", len(intents))
    st.metric("Training Patterns", total_patterns)
    
    st.markdown("---")
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Chat history cleared! How can I help you?"
        })
        st.rerun()
    
    st.markdown("---")
    st.caption("🔒 Secure Banking Support")
    st.caption("Built with ❤️ using Python & ML")

# Main chat interface
st.title("💬 SmartBank AI Support Chat")
st.markdown("*Powered by Natural Language Processing & Machine Learning*")
st.markdown("---")

# Load intents and initialize NLP model
intents = load_intents()
vectorizer, pattern_vectors, pattern_to_intent = initialize_nlp_model(intents)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "confidence" in message:
            st.caption(f"🎯 Confidence: {message['confidence']:.2%} | Intent: {message['intent']}")

# Chat input
if user_input := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            bot_response, confidence, intent = get_bot_response(
                user_input, 
                vectorizer, 
                pattern_vectors, 
                pattern_to_intent
            )
            st.markdown(bot_response)
            st.caption(f"🎯 Confidence: {confidence:.2%} | Intent: {intent}")
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response,
        "confidence": confidence,
        "intent": intent
    })

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("📞 **24/7 Helpline:** 1-800-SMART-BANK")
with col2:
    st.markdown("📧 **Email:** support@smartbank.com")
with col3:
    st.markdown("🌐 **Website:** www.smartbank.com")