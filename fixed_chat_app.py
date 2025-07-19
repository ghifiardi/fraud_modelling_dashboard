#!/usr/bin/env python3
"""
Fixed Chat App - Using OpenAI library for better authentication
"""

import streamlit as st
from openai import OpenAI
import time

# Configure OpenAI client
client = OpenAI(api_key="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")

st.set_page_config(
    page_title="Fixed Fraud Detection Chat",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Fixed Fraud Detection Chat")
st.markdown("Using OpenAI library v1.0+ for better authentication")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about fraud detection..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # System prompt for fraud detection context
            system_prompt = """You are an AI assistant for a bank fraud detection dashboard. 
            Help users understand fraud detection concepts, explain the dashboard features, 
            and provide insights about transaction monitoring and risk assessment. 
            Be helpful, informative, and professional."""
            
            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history (last 10 messages to avoid token limits)
            for msg in st.session_state.messages[-10:]:
                messages.append(msg)
            
            # Show "thinking" indicator
            with st.spinner("🤔 Thinking..."):
                # Make API call using new OpenAI client
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
            
            # Extract response
            full_response = response.choices[0].message.content
            
            # Stream the response
            full_response_placeholder = st.empty()
            for chunk in full_response.split():
                full_response_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            full_response_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"❌ **Error**: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with information
with st.sidebar:
    st.header("🔧 Debug Info")
    st.info(f"API Key: {client.api_key[:10]}...{client.api_key[-4:]}")
    st.info(f"Messages in history: {len(st.session_state.messages)}")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.header("💡 Suggested Questions")
    suggestions = [
        "What is fraud detection?",
        "How does the dashboard work?",
        "What are the risk levels?",
        "Explain transaction monitoring",
        "How do you detect suspicious transactions?",
        "What are common fraud patterns?",
        "How accurate is the fraud detection model?"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggest_{suggestion}"):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            st.rerun()

# Footer
st.markdown("---")
st.markdown("**Status**: Using OpenAI library v1.0+ for better authentication handling") 