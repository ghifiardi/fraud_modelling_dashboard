#!/usr/bin/env python3
"""
Debug Chat - Test OpenAI API Key Configuration
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.append('src')

st.set_page_config(
    page_title="Chat Debug",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Chat Debug - OpenAI API Key Test")

# Test API key access
st.header("1. API Key Access Test")

try:
    # Test different ways to access the API key
    st.subheader("Method 1: st.secrets.get()")
    try:
        api_key_1 = st.secrets.get('OPENAI_API_KEY')
        if api_key_1:
            st.success(f"✅ Found API key: {api_key_1[:10]}...{api_key_1[-4:]}")
        else:
            st.error("❌ No API key found in st.secrets")
    except Exception as e:
        st.error(f"❌ Error accessing st.secrets: {e}")
    
    st.subheader("Method 2: os.getenv()")
    api_key_2 = os.getenv('OPENAI_API_KEY')
    if api_key_2:
        st.success(f"✅ Found API key: {api_key_2[:10]}...{api_key_2[-4:]}")
    else:
        st.error("❌ No API key found in environment variables")
    
    st.subheader("Method 3: Direct st.secrets access")
    try:
        api_key_3 = st.secrets['OPENAI_API_KEY']
        st.success(f"✅ Found API key: {api_key_3[:10]}...{api_key_3[-4:]}")
    except Exception as e:
        st.error(f"❌ Error accessing st.secrets directly: {e}")
    
    st.subheader("Method 4: Check all secrets")
    try:
        all_secrets = dict(st.secrets)
        st.info(f"Available secrets: {list(all_secrets.keys())}")
        if 'OPENAI_API_KEY' in all_secrets:
            st.success("✅ OPENAI_API_KEY found in secrets")
        else:
            st.error("❌ OPENAI_API_KEY not found in secrets")
    except Exception as e:
        st.error(f"❌ Error listing secrets: {e}")

except Exception as e:
    st.error(f"❌ General error: {e}")

# Test OpenAI API call
st.header("2. OpenAI API Test")

if st.button("Test OpenAI API Call"):
    try:
        from src.llm_chatbot import LLMChatbotService
        
        chatbot = LLMChatbotService()
        
        st.info("Testing OpenAI API call...")
        
        # Test the API call directly
        response = chatbot._try_openai("Hello, this is a test message")
        
        if response and response != "[OpenAI API key not set]":
            st.success("✅ OpenAI API call successful!")
            st.info(f"Response: {response}")
        else:
            st.error(f"❌ OpenAI API call failed: {response}")
            
    except Exception as e:
        st.error(f"❌ Error testing OpenAI API: {e}")
        st.exception(e)

# Manual API key input for testing
st.header("3. Manual API Key Test")

manual_key = st.text_input("Enter OpenAI API Key manually for testing:", type="password")

if manual_key and st.button("Test Manual Key"):
    try:
        import requests
        
        headers = {
            "Authorization": f"Bearer {manual_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello, this is a test"}],
                "max_tokens": 50
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            st.success("✅ Manual API key works!")
            st.info(f"Response: {data['choices'][0]['message']['content']}")
        else:
            st.error(f"❌ API call failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"❌ Error with manual key: {e}")

# Instructions
st.header("4. How to Fix")

st.markdown("""
### If the API key is not working:

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Find your app** and click on it
3. **Click "Settings"** (gear icon)
4. **Click "Secrets"** in the left sidebar
5. **Add this configuration**:

```toml
OPENAI_API_KEY = "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"
```

6. **Click "Save"**
7. **Redeploy your app**

### Alternative: Use Environment Variables

If Streamlit secrets don't work, you can also set environment variables in Streamlit Cloud settings.
""") 