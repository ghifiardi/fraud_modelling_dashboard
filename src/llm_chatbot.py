#!/usr/bin/env python3
"""
LLM Chatbot Service with Multiple Fallback Options
Supports local Ollama, OpenAI, HuggingFace, and rule-based responses
"""

import streamlit as st
import requests
import os
import json
import time
from typing import Optional, Dict, Any

class LLMChatbotService:
    def __init__(self):
        # Try to get API keys from Streamlit secrets first (for cloud deployment)
        try:
            self.openai_api_key = st.secrets.get('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
            self.huggingface_api_key = st.secrets.get('HUGGINGFACE_API_KEY', os.getenv('HUGGINGFACE_API_KEY'))
        except:
            # Fallback to environment variables
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
    def get_response(self, user_input: str, chat_history: list = None) -> str:
        """
        Get response from LLM with multiple fallback options
        """
        # Try local Ollama first
        response = self._try_ollama(user_input, chat_history)
        if response:
            return response
            
        # Try OpenAI if API key is available
        if self.openai_api_key:
            response = self._try_openai(user_input, chat_history)
            if response:
                return response
                
        # Try HuggingFace Inference API
        response = self._try_huggingface(user_input, chat_history)
        if response:
            return response
            
        # Fallback to rule-based responses
        return self._rule_based_response(user_input)
    
    def _try_ollama(self, user_input: str, chat_history: list = None) -> Optional[str]:
        """Try local Ollama server"""
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "mistral:7b-instruct",
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": user_input}
                    ],
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            return None
    
    def _try_openai(self, user_input: str, chat_history: list = None) -> Optional[str]:
        """Try OpenAI API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_input}
            ]
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.7
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return None
    
    def _try_huggingface(self, user_input: str, chat_history: list = None) -> Optional[str]:
        """Try HuggingFace Inference API"""
        if not self.huggingface_api_key:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.huggingface_api_key}",
                "Content-Type": "application/json"
            }
            
            # Use a simpler, more reliable model
            payload = {
                "inputs": f"Question: {user_input}\nAnswer:",
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            # Try multiple models in case one is unavailable
            models = [
                "microsoft/DialoGPT-medium",
                "gpt2",
                "distilgpt2"
            ]
            
            for model in models:
                try:
                    response = requests.post(
                        f"https://api-inference.huggingface.co/models/{model}",
                        headers=headers,
                        json=payload,
                        timeout=15
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list) and len(data) > 0:
                            generated_text = data[0].get("generated_text", "")
                            # Extract the generated part
                            if "Answer:" in generated_text:
                                return generated_text.split("Answer:")[-1].strip()
                            else:
                                return generated_text.strip()
                except Exception:
                    continue
                    
            return None
        except Exception as e:
            return None
    
    def _rule_based_response(self, user_input: str) -> str:
        """Rule-based fallback responses"""
        user_input_lower = user_input.lower()
        
        # Common fraud detection questions
        if any(word in user_input_lower for word in ['fraud', 'detection', 'what is']):
            return """Fraud detection is a system that identifies suspicious or fraudulent transactions in real-time. Our dashboard uses machine learning models to analyze transaction patterns and flag potential risks. The system considers factors like transaction amount, location, time, and customer behavior to make predictions."""
        
        elif any(word in user_input_lower for word in ['dashboard', 'how to use', 'help']):
            return """The dashboard has 5 main sections:
1. **Real-time Dashboard**: Live metrics and transaction monitoring
2. **Transaction Monitor**: Detailed transaction analysis
3. **Analytics**: Charts and performance metrics
4. **Model Management**: Model status and configuration
5. **Alerts & Logs**: Risk alerts and transaction logs

You can navigate between tabs to explore different features."""
        
        elif any(word in user_input_lower for word in ['risk', 'level', 'high', 'medium', 'low']):
            return """Risk levels in our system:
- **HIGH RISK**: Top 5% risk scores - Transactions are blocked
- **MEDIUM RISK**: Top 15% risk scores - Additional verification required
- **LOW RISK**: Top 30% risk scores - Monitor closely
- **SAFE**: Below 30th percentile - Allow transaction

The system automatically categorizes transactions based on ML model predictions."""
        
        elif any(word in user_input_lower for word in ['model', 'performance', 'accuracy']):
            return """Our fraud detection system uses multiple ML models:
- **Logistic Regression**: Baseline model with interpretability
- **Random Forest**: Robust ensemble method
- **Isolation Forest**: Anomaly detection

Performance metrics include AUC-ROC, Precision, Recall, and F1-Score. You can view detailed metrics in the Analytics tab."""
        
        elif any(word in user_input_lower for word in ['transaction', 'monitor', 'real-time']):
            return """Real-time transaction monitoring shows:
- Live transaction feed with risk scores
- Transaction volume over time
- Fraud detection rates
- Customer risk profiles
- Performance metrics

The system processes transactions in real-time and updates the dashboard automatically."""
        
        elif any(word in user_input_lower for word in ['hello', 'hi', 'hey']):
            return """Hello! I'm your AI assistant for the fraud detection dashboard. I can help you understand how the system works, explain model predictions, and guide you through the dashboard features. What would you like to know about fraud detection?"""
        
        else:
            return """I'm here to help you with fraud detection questions! You can ask me about:
- How fraud detection works
- Understanding the dashboard
- Risk levels and thresholds
- Model performance
- Transaction monitoring
- Or any other fraud detection topics

What would you like to know?"""
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for LLM interactions"""
        return """You are a helpful AI assistant for a fraud detection dashboard. You help users understand:

1. **Fraud Detection Concepts**: Explain how fraud detection works, what factors are considered, and how ML models make predictions.

2. **Dashboard Navigation**: Guide users through the different sections (Real-time Dashboard, Transaction Monitor, Analytics, Model Management, Alerts & Logs).

3. **Risk Assessment**: Explain risk levels (HIGH, MEDIUM, LOW, SAFE) and what they mean for transaction processing.

4. **Model Performance**: Help users understand metrics like AUC-ROC, Precision, Recall, and F1-Score.

5. **Technical Details**: Explain the machine learning models used (Logistic Regression, Random Forest, Isolation Forest) and their roles.

6. **Best Practices**: Provide guidance on using the dashboard effectively and interpreting results.

Keep responses concise, helpful, and focused on fraud detection. If you don't know something specific about the system, suggest where users can find more information in the dashboard."""

def create_chatbot_ui():
    """Create the chatbot UI in Streamlit sidebar"""
    st.sidebar.header("💬 AI Chatbot Assistant")
    
    # Initialize chatbot service
    if 'chatbot_service' not in st.session_state:
        st.session_state.chatbot_service = LLMChatbotService()
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Show connection status
    with st.sidebar.expander("🔗 Connection Status", expanded=False):
        if st.session_state.chatbot_service.openai_api_key:
            st.success("✅ OpenAI API available")
        else:
            st.info("ℹ️ OpenAI API key not set (optional)")
        
        if st.session_state.chatbot_service.huggingface_api_key:
            st.success("✅ HuggingFace API available")
        else:
            st.info("ℹ️ HuggingFace API key not set (optional)")
    
    # Chat input
    user_input = st.sidebar.text_input("Ask me anything about fraud detection...")
    
    if user_input:
        with st.sidebar.spinner("🤖 Thinking..."):
            # Get response from chatbot service
            response = st.session_state.chatbot_service.get_response(user_input, st.session_state.chat_history)
            
            # Add to chat history
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", response))
    
    # Display chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.sidebar.markdown(f"**You:** {msg}")
        else:
            st.sidebar.markdown(f"**Assistant:** {msg}")
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Help button
    if st.sidebar.button("❓ Help"):
        st.sidebar.info("""
        **How to use the chatbot:**
        
        Ask questions about:
        - Fraud detection concepts
        - Dashboard navigation
        - Risk levels and thresholds
        - Model performance
        - Transaction monitoring
        
        The chatbot will automatically use the best available LLM service.
        """) 