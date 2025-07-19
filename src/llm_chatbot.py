#!/usr/bin/env python3
"""
LLM Chatbot Service with Multiple Fallback Options and FraudLabs Pro Integration
Supports local Ollama, OpenAI, HuggingFace, and real-time fraud screening
"""

import streamlit as st
import requests
import os
import json
import time
import re
from typing import Optional, Dict, Any

class LLMChatbotService:
    def __init__(self):
        # Try to get API keys from Streamlit secrets first (for cloud deployment)
        try:
            self.openai_api_key = st.secrets.get('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
            self.huggingface_api_key = st.secrets.get('HUGGINGFACE_API_KEY', os.getenv('HUGGINGFACE_API_KEY'))
            self.fraudlabs_api_key = st.secrets.get('FRAUDLABS_API_KEY', os.getenv('FRAUDLABS_API_KEY', 'TNFUSCVIQFJEV4QYO10B7EONML4515EP'))
        except:
            # Fallback to environment variables
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
            self.fraudlabs_api_key = os.getenv('FRAUDLABS_API_KEY', 'TNFUSCVIQFJEV4QYO10B7EONML4515EP')
        
        self.selected_model = st.session_state.get('llm_model', 'openai')
        
    def get_response(self, user_input: str, chat_history: list = []) -> str:
        """
        Get response from LLM with multiple fallback options and FraudLabs Pro integration
        """
        # Check if this is a FraudLabs Pro query first
        fraudlabs_response = self._check_fraudlabs_query(user_input)
        if fraudlabs_response:
            return fraudlabs_response
            
        # Model selection logic
        model = st.session_state.get('llm_model', 'openai')
        if model == 'openai':
            response = self._try_openai(user_input, chat_history)
            if response:
                return response
        elif model == 'huggingface':
            response = self._try_huggingface(user_input, chat_history)
            if response:
                return response
        elif model == 'ollama':
            response = self._try_ollama(user_input, chat_history)
            if response:
                return response
        # Fallback to rule-based responses
        return self._rule_based_response(user_input)
    
    def _check_fraudlabs_query(self, user_input: str) -> Optional[str]:
        """Check if user input is asking for FraudLabs Pro screening"""
        user_input_lower = user_input.lower()
        
        # Keywords that indicate FraudLabs Pro screening request
        fraudlabs_keywords = [
            'screen transaction', 'fraud check', 'risk score', 'fraudlabs',
            'screen order', 'transaction risk', 'fraud screening',
            'check ip', 'check email', 'check amount'
        ]
        
        if any(keyword in user_input_lower for keyword in fraudlabs_keywords):
            return self._get_fraudlabs_screening_help()
        
        # Check if user provided transaction details
        if any(word in user_input_lower for word in ['ip:', 'email:', 'amount:', 'transaction:']):
            return self._extract_and_screen_transaction(user_input)
            
        return None
    
    def _get_fraudlabs_screening_help(self) -> str:
        """Provide help for FraudLabs Pro screening"""
        return """**FraudLabs Pro Real-Time Screening** 🔍

To screen a transaction for fraud risk, provide the details in this format:
```
IP: 1.2.3.4
Email: customer@example.com  
Amount: 250.00
```

Or ask: "Screen this transaction: IP 1.2.3.4, Email customer@example.com, Amount $250"

**What FraudLabs Pro checks:**
- IP address risk (proxy, VPN, location)
- Email risk (disposable, free email, domain age)
- Amount patterns and velocity
- Device fingerprinting
- Geographic risk factors
- Blacklist checking

**Response includes:**
- Risk score (0-100)
- Status (APPROVE/REVIEW/DECLINE)
- Risk factors and explanations
- Recommended actions

Try it now with a transaction!"""
    
    def _extract_and_screen_transaction(self, user_input: str) -> str:
        """Extract transaction details and screen with FraudLabs Pro"""
        try:
            # Extract IP, email, and amount from user input
            ip_address = None
            email = None
            amount = None
            
            # Simple extraction logic
            lines = user_input.split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'ip:' in line_lower or 'ip ' in line_lower:
                    ip_address = line.split(':')[-1].strip().split()[0]
                elif 'email:' in line_lower or 'email ' in line_lower:
                    email = line.split(':')[-1].strip().split()[0]
                elif 'amount:' in line_lower or 'amount ' in line_lower or '$' in line:
                    amount_str = line.split(':')[-1].strip().split()[0]
                    amount = amount_str.replace('$', '').replace(',', '')
            
            # If not found in structured format, try to extract from text
            if not ip_address or not email or not amount:
                # Fallback extraction
                ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                amount_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
                
                ip_match = re.search(ip_pattern, user_input)
                email_match = re.search(email_pattern, user_input)
                amount_match = re.search(amount_pattern, user_input)
                
                if ip_match:
                    ip_address = ip_match.group()
                if email_match:
                    email = email_match.group()
                if amount_match:
                    amount = amount_match.group(1).replace(',', '')
            
            if ip_address and email and amount:
                return self._screen_transaction_fraudlabs(ip_address, email, amount)
            else:
                return """**Missing transaction details** ❌

Please provide:
- IP address (e.g., 1.2.3.4)
- Email address (e.g., customer@example.com)
- Amount (e.g., 250.00)

Format: "Screen transaction: IP 1.2.3.4, Email customer@example.com, Amount $250"
"""
        except Exception as e:
            return f"Error processing transaction details: {str(e)}"
    
    def _screen_transaction_fraudlabs(self, ip_address: str, email: str, amount: str) -> str:
        """Screen transaction using FraudLabs Pro API"""
        try:
            url = "https://api.fraudlabspro.com/v1/order/screen"
            params = {
                "key": self.fraudlabs_api_key,
                "ip_address": ip_address,
                "email": email,
                "amount": amount,
                "currency": "USD"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Format the response
            risk_score = data.get('risk_score', 'N/A')
            status = data.get('status', 'UNKNOWN')
            risk_factors = data.get('risk_factors', [])
            
            result = f"""**FraudLabs Pro Screening Result** 🔍

**Transaction Details:**
- IP: {ip_address}
- Email: {email}
- Amount: ${amount}

**Risk Assessment:**
- **Risk Score**: {risk_score}/100
- **Status**: {status}
- **Recommendation**: {self._get_fraudlabs_recommendation(status)}

**Risk Factors:**
"""
            
            if risk_factors:
                for factor in risk_factors:
                    result += f"- {factor}\n"
            else:
                result += "- No specific risk factors identified\n"
            
            result += f"""
**Response Time**: {response.elapsed.total_seconds():.2f}s
**API Status**: ✅ Live data from FraudLabs Pro

This is real-time fraud screening data, not AI-generated content!"""
            
            return result
            
        except requests.exceptions.RequestException as e:
            return f"""**FraudLabs Pro API Error** ❌

Error: {str(e)}

This could be due to:
- Network connectivity issues
- API rate limiting
- Invalid API key
- Service temporarily unavailable

Please try again in a few moments."""
        except Exception as e:
            return f"Error screening transaction: {str(e)}"
    
    def _get_fraudlabs_recommendation(self, status: str) -> str:
        """Get recommendation based on FraudLabs Pro status"""
        recommendations = {
            'APPROVE': '✅ Allow transaction',
            'REVIEW': '⚠️ Review manually',
            'DECLINE': '❌ Block transaction',
            'UNKNOWN': '❓ Manual review recommended'
        }
        return recommendations.get(status, 'Manual review recommended')
    
    def _try_ollama(self, user_input: str, chat_history: list = []) -> Optional[str]:
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
    
    def _try_openai(self, user_input: str, chat_history: list = []) -> Optional[str]:
        """Try OpenAI API"""
        if not self.openai_api_key:
            return "[OpenAI API key not set]"
            
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
    
    def _try_huggingface(self, user_input: str, chat_history: list = []) -> Optional[str]:
        """Try HuggingFace Inference API"""
        if not self.huggingface_api_key:
            return "[HuggingFace API key not set]"
            
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
            return "[HuggingFace: No response from available models]"
        except Exception as e:
            return f"[HuggingFace error] {e}"
    
    def _rule_based_response(self, user_input: str) -> str:
        """Rule-based fallback responses"""
        user_input_lower = user_input.lower()
        
        # FraudLabs Pro specific responses
        if any(word in user_input_lower for word in ['fraudlabs', 'fraud labs', 'screening']):
            return self._get_fraudlabs_screening_help()
        
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
            return """Hello! I'm your AI assistant for the fraud detection dashboard. I can help you understand how the system works, explain model predictions, and guide you through the dashboard features. 

**New Feature**: I can also screen transactions in real-time using FraudLabs Pro! Just ask me to screen a transaction with IP, email, and amount details.

What would you like to know about fraud detection?"""
        
        else:
            return """I'm here to help you with fraud detection questions! You can ask me about:
- How fraud detection works
- Understanding the dashboard
- Risk levels and thresholds
- Model performance
- Transaction monitoring
- **Real-time fraud screening with FraudLabs Pro**

What would you like to know?"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        splunk_mode = st.session_state.get('splunk_mode', False)
        
        if splunk_mode:
            return """You are a Splunk fraud analytics expert. Provide detailed, practical guidance on:
- Splunk fraud detection implementation
- SPL (Search Processing Language) queries
- Risk-based alerting and correlation searches
- Machine Learning Toolkit (MLTK) for fraud
- Enterprise Security (ES) integration
- SOAR playbooks and automation
- Data onboarding and indexing strategies

Always provide actionable, specific advice with examples."""
        else:
            return """You are an AI assistant for a bank fraud detection dashboard. Help users understand:
- Fraud detection concepts and methodologies
- Dashboard features and navigation
- Risk assessment and scoring
- Model performance and interpretation
- Real-time monitoring capabilities
- Transaction analysis and alerts

Provide clear, helpful responses focused on fraud detection and banking security."""

def create_chatbot_ui():
    """Create the chatbot UI in Streamlit sidebar, with Splunk Expert mode toggle"""
    st.sidebar.header("💬 AI Chatbot Assistant")
    
    # Splunk Expert Mode toggle
    splunk_mode = st.sidebar.toggle("Splunk Expert Mode", value=st.session_state.get('splunk_mode', False))
    st.session_state['splunk_mode'] = splunk_mode
    
    # Model selection (remove Cisco Foundation)
    llm_model = st.sidebar.selectbox(
        "Choose LLM Model",
        ["openai", "huggingface", "ollama"],
        format_func=lambda x: {
            "openai": "OpenAI (GPT-3.5/4)",
            "huggingface": "HuggingFace (General)",
            "ollama": "Local Ollama (Mistral)"
        }[x],
        index=0
    )
    st.session_state['llm_model'] = llm_model
    
    # FraudLabs Pro integration info
    st.sidebar.info("🔍 **FraudLabs Pro Integration**: Ask me to screen transactions for real-time fraud risk!")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about fraud detection or screen a transaction..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chatbot = LLMChatbotService()
                response = chatbot.get_response(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun() 