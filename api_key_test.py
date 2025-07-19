#!/usr/bin/env python3
"""
API Key Test - Verify the API key format and validity
"""

import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="API Key Test",
    page_icon="🔑",
    layout="wide"
)

st.title("🔑 API Key Test")

# Test different API key formats
api_keys = {
    "Current Key": "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",
    "Standard Format": "sk-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    "Test Key": "sk-test1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
}

st.header("1. API Key Analysis")

for name, key in api_keys.items():
    st.subheader(f"{name}")
    st.info(f"Length: {len(key)} characters")
    st.info(f"Starts with: {key[:10]}")
    st.info(f"Ends with: {key[-4:]}")
    st.info(f"Format valid: {key.startswith('sk-')}")
    
    if st.button(f"Test {name}", key=f"test_{name}"):
        try:
            client = OpenAI(api_key=key)
            
            # Try a simple API call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say hello!"}],
                max_tokens=10
            )
            
            st.success(f"✅ {name} works! Response: {response.choices[0].message.content}")
            
        except Exception as e:
            st.error(f"❌ {name} failed: {str(e)}")

st.header("2. API Key Format Guide")

st.markdown("""
**Valid API Key Formats:**
- `sk-` followed by 48 characters (standard OpenAI key)
- `sk-proj-` followed by characters (project-specific key)

**Your Current Key Analysis:**
- Length: 164 characters (unusually long)
- Format: `sk-proj-` (project key format)
- Issue: Likely invalid or expired

**Solutions:**
1. **Get a new API key** from https://platform.openai.com/account/api-keys
2. **Use a standard format key** (sk- + 48 characters)
3. **Check if the project key is still valid**
""")

st.header("3. Quick Fix Options")

if st.button("Create Working Demo with Fallback"):
    st.success("""
    **Option 1: Use a working demo key**
    - I can provide a temporary working key for testing
    
    **Option 2: Create fallback responses**
    - Build intelligent responses without API calls
    
    **Option 3: Guide you to get a new key**
    - Help you create a proper OpenAI API key
    """)

st.header("4. Next Steps")

st.markdown("""
**Immediate Actions:**
1. **Get a new API key** from OpenAI platform
2. **Test the new key** with this app
3. **Update the chat app** with the working key

**Alternative:**
- Use a different AI service (HuggingFace, local models)
- Create rule-based responses for demo purposes
""") 