#!/usr/bin/env python3
"""
Direct API Test - See exactly what's happening
"""

import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Direct API Test",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Direct OpenAI API Test")

# API key
api_key = "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"

st.info(f"Using API key: {api_key[:10]}...{api_key[-4:]}")

# Test 1: Simple API call
st.header("1. Simple API Call Test")

if st.button("Test Simple API Call"):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Say hello!"}],
            "max_tokens": 50
        }
        
        st.info("Making API call...")
        st.json(payload)
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        st.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            st.success("✅ API call successful!")
            st.json(data)
            st.info(f"Response: {data['choices'][0]['message']['content']}")
        else:
            st.error(f"❌ API call failed: {response.status_code}")
            st.error(f"Error: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Exception: {str(e)}")
        st.exception(e)

# Test 2: Fraud detection specific call
st.header("2. Fraud Detection API Call Test")

if st.button("Test Fraud Detection Call"):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are an AI assistant for a bank fraud detection dashboard. Help users understand fraud detection concepts."""
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is fraud detection?"}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        st.info("Making fraud detection API call...")
        st.json(payload)
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        st.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            st.success("✅ Fraud detection API call successful!")
            st.json(data)
            st.info(f"Response: {data['choices'][0]['message']['content']}")
        else:
            st.error(f"❌ API call failed: {response.status_code}")
            st.error(f"Error: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Exception: {str(e)}")
        st.exception(e)

# Test 3: Check API key validity
st.header("3. API Key Validation")

if st.button("Validate API Key"):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Try to list models to validate the key
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=30
        )
        
        st.info(f"Validation response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            st.success("✅ API key is valid!")
            st.info(f"Available models: {len(data['data'])}")
            model_names = [model['id'] for model in data['data'][:5]]
            st.info(f"Sample models: {model_names}")
        else:
            st.error(f"❌ API key validation failed: {response.status_code}")
            st.error(f"Error: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Exception: {str(e)}")
        st.exception(e)

# Test 4: Network connectivity
st.header("4. Network Connectivity Test")

if st.button("Test Network"):
    try:
        # Test basic connectivity
        response = requests.get("https://api.openai.com/v1/models", timeout=10)
        st.info(f"OpenAI API connectivity: {response.status_code}")
        
        if response.status_code == 401:
            st.warning("⚠️ Can reach OpenAI API but authentication failed (expected without key)")
        elif response.status_code == 200:
            st.success("✅ Can reach OpenAI API")
        else:
            st.error(f"❌ Cannot reach OpenAI API: {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Network error: {str(e)}")

# Debug information
st.header("5. Debug Information")

st.markdown(f"""
**API Key Info:**
- Length: {len(api_key)} characters
- Starts with: {api_key[:10]}
- Ends with: {api_key[-4:]}
- Format: {api_key.startswith('sk-')}

**Environment:**
- Streamlit Cloud: Yes
- Python version: {st.__version__}
- Requests available: {requests.__version__}

**Common Issues:**
1. **API key format**: Should start with 'sk-'
2. **Network connectivity**: Check if OpenAI API is reachable
3. **Rate limiting**: Check if we're hitting rate limits
4. **Model availability**: Check if gpt-3.5-turbo is available
""") 