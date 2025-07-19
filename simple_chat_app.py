#!/usr/bin/env python3
"""
Simple Chat App - Test OpenAI API Integration
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.append('src')

st.set_page_config(
    page_title="AI Fraud Detection Chat",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.user-message {
    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    color: white;
}
.assistant-message {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🛡️ AI Fraud Detection Chat</h1>', unsafe_allow_html=True)

# Status indicator
st.success("✅ Chat system ready!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about fraud detection..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        try:
            # Try to use the chatbot service
            from llm_chatbot import LLMChatbotService
            
            chatbot = LLMChatbotService()
            response = chatbot.get_response(prompt)
            
            if response and response != "[OpenAI API key not set]":
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Fallback response
                fallback_response = """I'm here to help you with fraud detection! 

**What I can help with:**
- Explain fraud detection concepts
- Guide you through the dashboard
- Answer questions about risk levels
- Help with transaction analysis
- Provide fraud screening information

**Try asking:**
- "What is fraud detection?"
- "How does the dashboard work?"
- "What are the risk levels?"
- "Explain transaction monitoring"

The OpenAI API integration is being configured. For now, I'm using my built-in knowledge to help you!"""
                
                st.markdown(fallback_response)
                st.session_state.messages.append({"role": "assistant", "content": fallback_response})
                
        except Exception as e:
            error_response = f"""I encountered an error: {str(e)}

**What you can do:**
1. Check if the OpenAI API key is configured in Streamlit Cloud
2. Try asking a simple question about fraud detection
3. Contact support if the issue persists

**Example questions:**
- "What is fraud detection?"
- "How does the system work?"
- "What are the risk levels?" """
            
            st.markdown(error_response)
            st.session_state.messages.append({"role": "assistant", "content": error_response})

# Sidebar with information
st.sidebar.header("💬 Chat Information")

st.sidebar.markdown("""
**About this chat:**
- Powered by AI for fraud detection assistance
- Can answer questions about the dashboard
- Provides real-time fraud screening
- Helps with transaction analysis

**API Status:**
- OpenAI integration: Configuring
- Fallback responses: Available
- Real-time screening: Ready
""")

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Test API button
if st.sidebar.button("Test API Connection"):
    try:
        from llm_chatbot import LLMChatbotService
        chatbot = LLMChatbotService()
        
        # Test the API
        test_response = chatbot._try_openai("Hello, this is a test")
        
        if test_response and test_response != "[OpenAI API key not set]":
            st.sidebar.success("✅ OpenAI API working!")
        else:
            st.sidebar.error("❌ OpenAI API not configured")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")

# Instructions
st.sidebar.markdown("""
**To enable OpenAI:**
1. Go to Streamlit Cloud settings
2. Add to Secrets:
```toml
OPENAI_API_KEY = "your-api-key-here"
```
3. Save and redeploy
""") 