#!/usr/bin/env python3
"""
Test OpenAI API Key Access
"""

import streamlit as st
import requests
import os

st.set_page_config(
    page_title="API Key Test",
    page_icon="🔑",
    layout="wide"
)

st.title("🔑 OpenAI API Key Test")

# Test 1: Check if API key is accessible
st.header("1. API Key Access Test")

# Method 1: st.secrets
st.subheader("Method 1: st.secrets")
try:
    api_key_1 = st.secrets.get('OPENAI_API_KEY')
    if api_key_1:
        st.success(f"✅ Found in st.secrets: {api_key_1[:10]}...{api_key_1[-4:]}")
    else:
        st.error("❌ Not found in st.secrets")
except Exception as e:
    st.error(f"❌ Error accessing st.secrets: {e}")

# Method 2: Direct access
st.subheader("Method 2: Direct st.secrets access")
try:
    api_key_2 = st.secrets['OPENAI_API_KEY']
    st.success(f"✅ Found directly: {api_key_2[:10]}...{api_key_2[-4:]}")
except Exception as e:
    st.error(f"❌ Error with direct access: {e}")

# Method 3: Environment variable
st.subheader("Method 3: Environment variable")
api_key_3 = os.getenv('OPENAI_API_KEY')
if api_key_3:
    st.success(f"✅ Found in env: {api_key_3[:10]}...{api_key_3[-4:]}")
else:
    st.error("❌ Not found in environment variables")

# Test 2: List all secrets
st.header("2. All Available Secrets")
try:
    all_secrets = dict(st.secrets)
    st.info(f"Available secrets: {list(all_secrets.keys())}")
    for key, value in all_secrets.items():
        if 'API' in key.upper() or 'KEY' in key.upper():
            st.write(f"  {key}: {str(value)[:10]}...{str(value)[-4:]}")
        else:
            st.write(f"  {key}: {value}")
except Exception as e:
    st.error(f"❌ Error listing secrets: {e}")

# Test 3: Direct API call
st.header("3. Direct OpenAI API Test")

# Get the API key
api_key = None
if st.secrets.get('OPENAI_API_KEY'):
    api_key = st.secrets['OPENAI_API_KEY']
elif os.getenv('OPENAI_API_KEY'):
    api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    st.success(f"✅ Using API key: {api_key[:10]}...{api_key[-4:]}")
    
    if st.button("Test OpenAI API Call"):
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Say hello and confirm you're working!"}],
                    "max_tokens": 50
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success("✅ OpenAI API call successful!")
                st.info(f"Response: {data['choices'][0]['message']['content']}")
            else:
                st.error(f"❌ API call failed: {response.status_code}")
                st.error(f"Error: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Error with API call: {e}")
else:
    st.error("❌ No API key found!")

# Test 4: Manual API key input
st.header("4. Manual API Key Test")

manual_key = st.text_input("Enter API key manually:", type="password")

if manual_key and st.button("Test Manual Key"):
    try:
        headers = {
            "Authorization": f"Bearer {manual_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello! Are you working?"}],
                "max_tokens": 50
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            st.success("✅ Manual API key works!")
            st.info(f"Response: {data['choices'][0]['message']['content']}")
        else:
            st.error(f"❌ Manual API call failed: {response.status_code}")
            st.error(f"Error: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Error with manual key: {e}")

# Instructions
st.header("5. How to Fix")

st.markdown("""
### If the API key is not working:

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Find your app** and click on it
3. **Click "Settings"** (gear icon)
4. **Click "Secrets"** in the left sidebar
5. **Make sure you have this exact configuration**:

```toml
OPENAI_API_KEY = "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"
```

6. **Click "Save changes"**
7. **Wait 1-2 minutes** for changes to propagate
8. **Redeploy your app**

### Common Issues:
- **Wrong format**: Make sure to use quotes around the API key
- **Extra spaces**: Remove any extra spaces before or after the key
- **Wrong variable name**: Must be exactly `OPENAI_API_KEY`
- **Not saved**: Make sure to click "Save changes"
""") 