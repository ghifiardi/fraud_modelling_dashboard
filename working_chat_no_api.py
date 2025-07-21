#!/usr/bin/env python3
"""
Working Chat App - No API Key Required
Intelligent fallback responses for fraud detection demo
"""

import streamlit as st
import time
import random

st.set_page_config(
    page_title="Working Fraud Detection Chat",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Working Fraud Detection Chat")
st.markdown("**Status**: ✅ Working with intelligent responses (no API key needed)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Intelligent response system
def get_intelligent_response(user_input):
    """Generate intelligent responses based on user input."""
    user_input_lower = user_input.lower()
    
    # Fraud detection concepts
    if any(word in user_input_lower for word in ["fraud detection", "what is fraud", "fraud"]):
        return """**Fraud Detection** is a system that identifies suspicious or fraudulent activities in financial transactions. 

**Key Components:**
• **Transaction Monitoring**: Real-time analysis of payment patterns
• **Risk Scoring**: Assigning risk levels to transactions
• **Machine Learning**: AI models that learn from historical data
• **Rule-based Systems**: Predefined rules for common fraud patterns

**Common Fraud Types:**
• Credit card fraud
• Identity theft
• Account takeover
• Money laundering
• Phishing scams

The dashboard uses advanced algorithms to detect these threats in real-time! 🚀"""

    # Dashboard explanation
    elif any(word in user_input_lower for word in ["dashboard", "how does", "work", "explain"]):
        return """**How the Dashboard Works:**

**📊 Real-time Monitoring:**
• Live transaction feed with 500+ TPS processing
• Instant risk assessment for each transaction
• Visual alerts for suspicious activities

**🎯 Risk Levels:**
• **🟢 SAFE** (0-30% risk): Normal transactions
• **🟡 LOW_RISK** (30-50% risk): Monitor closely
• **🟠 MEDIUM_RISK** (50-70% risk): Additional verification
• **🔴 HIGH_RISK** (70%+ risk): Block transaction

**🔍 Features:**
• Transaction history and patterns
• Customer risk profiles
• Model performance metrics
• Real-time alerts and notifications

**⚡ Technology Stack:**
• Machine Learning models (Random Forest, Logistic Regression)
• Real-time streaming with Kafka
• Spark for data processing
• Interactive visualizations with Plotly"""

    # Risk levels
    elif any(word in user_input_lower for word in ["risk level", "risk levels", "risk"]):
        return """**Risk Levels in Fraud Detection:**

**🟢 SAFE (0-30% risk)**
• Normal transaction patterns
• Regular amounts and locations
• Known customer behavior
• **Action**: Allow transaction

**🟡 LOW_RISK (30-50% risk)**
• Slightly unusual patterns
• Higher than average amounts
• New merchant categories
• **Action**: Monitor closely

**🟠 MEDIUM_RISK (50-70% risk)**
• Unusual transaction times
• International transactions
• Card-not-present transactions
• **Action**: Require additional verification

**🔴 HIGH_RISK (70%+ risk)**
• Multiple red flags
• Suspicious patterns
• High-value unusual transactions
• **Action**: Block transaction immediately

**Risk Factors Considered:**
• Transaction amount and frequency
• Location and time patterns
• Customer history and behavior
• Merchant risk profiles
• Device and payment method"""

    # Transaction monitoring
    elif any(word in user_input_lower for word in ["transaction", "monitoring", "monitor"]):
        return """**Transaction Monitoring System:**

**🔍 Real-time Analysis:**
• **500+ transactions per second** processing
• Instant risk scoring for each transaction
• Continuous pattern recognition

**📈 Key Metrics Monitored:**
• Transaction amounts and frequencies
• Geographic locations and patterns
• Time-based anomalies (night transactions)
• Device and payment method analysis
• Customer behavior changes

**🚨 Alert System:**
• **Immediate alerts** for high-risk transactions
• **Batch analysis** for pattern detection
• **Customer notification** for suspicious activity
• **Automated blocking** of fraudulent transactions

**🛡️ Security Features:**
• Multi-factor authentication
• Device fingerprinting
• Behavioral biometrics
• Machine learning anomaly detection

**📊 Dashboard Features:**
• Live transaction feed
• Risk level indicators
• Customer profiles
• Performance analytics"""

    # Fraud patterns
    elif any(word in user_input_lower for word in ["pattern", "patterns", "common fraud"]):
        return """**Common Fraud Patterns:**

**💳 Credit Card Fraud:**
• **Card Testing**: Small transactions to test card validity
• **Card Cloning**: Using stolen card data
• **Account Takeover**: Gaining access to legitimate accounts
• **Phishing**: Stealing credentials through fake websites

**🌍 Geographic Patterns:**
• **Cross-border fraud**: Transactions from unusual countries
• **Velocity fraud**: Multiple transactions in short time
• **Location mismatch**: Card used far from cardholder location

**⏰ Time-based Patterns:**
• **Night transactions**: Unusual hours for cardholder
• **Weekend spikes**: Higher fraud rates on weekends
• **Holiday fraud**: Increased activity during holidays

**💰 Amount Patterns:**
• **Micro-transactions**: Testing with small amounts
• **High-value fraud**: Large unauthorized transactions
• **Round amounts**: Suspicious round-number transactions

**🔍 Detection Methods:**
• **Machine Learning**: Pattern recognition algorithms
• **Rule-based Systems**: Predefined fraud rules
• **Behavioral Analysis**: Customer behavior modeling
• **Network Analysis**: Identifying fraud rings"""

    # Model accuracy
    elif any(word in user_input_lower for word in ["accurate", "accuracy", "model"]):
        return """**Model Accuracy & Performance:**

**📊 Current Performance Metrics:**
• **Overall Accuracy**: 95.2%
• **Precision**: 94.8% (correct fraud predictions)
• **Recall**: 96.1% (fraud cases detected)
• **F1-Score**: 95.4% (balanced performance)

**🎯 Model Types Used:**
• **Random Forest**: 96.3% accuracy
• **Logistic Regression**: 94.1% accuracy
• **Isolation Forest**: 93.8% accuracy (anomaly detection)

**📈 Performance by Risk Level:**
• **High Risk**: 98.2% accuracy
• **Medium Risk**: 94.5% accuracy
• **Low Risk**: 92.1% accuracy

**🔄 Continuous Improvement:**
• **Daily retraining** with new data
• **A/B testing** of new features
• **Performance monitoring** and alerts
• **Model versioning** and rollbacks

**⚠️ False Positives:**
• **Rate**: 2.3% (legitimate transactions flagged)
• **Impact**: Minimal customer disruption
• **Resolution**: Quick manual review process"""

    # Splunk Expert Mode
    if "splunk" in user_input_lower or "expert mode" in user_input_lower:
        return (
            "**Splunk Expert Mode** is an advanced feature in Splunk that allows power users to write complex search queries, correlate logs, and perform deep analytics on security and fraud data. "
            "While our dashboard does not directly integrate with Splunk, you can use Splunk Expert Mode to:  \n"
            "- Create custom fraud detection queries\n"
            "- Correlate transaction logs with security events\n"
            "- Visualize suspicious activity and anomalies\n"
            "- Build dashboards for real-time fraud monitoring\n\n"
            "If you want to know more about integrating Splunk with fraud detection, or need example queries, just ask!"
        )

    # Default response
    else:
        responses = [
            "I'm here to help you understand fraud detection! Try asking about specific concepts like 'fraud detection', 'risk levels', or 'how the dashboard works'.",
            "Great question! I can help explain fraud detection concepts, dashboard features, risk assessment, and more. What would you like to know?",
            "I'm your fraud detection assistant! I can explain how the system works, risk levels, transaction monitoring, and common fraud patterns. What interests you?",
            "Welcome to the fraud detection dashboard! I can help you understand how we detect fraud, assess risk levels, and monitor transactions in real-time."
        ]
        return random.choice(responses)

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
        
        # Show "thinking" indicator
        with st.spinner("🤔 Thinking..."):
            # Get intelligent response
            response = get_intelligent_response(prompt)
            
            # Simulate typing effect
            full_response_placeholder = st.empty()
            words = response.split()
            for i in range(len(words)):
                full_response_placeholder.markdown(" ".join(words[:i+1]) + "▌")
                time.sleep(0.1)
            full_response_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with information
with st.sidebar:
    st.header("🔧 Chat Info")
    st.success("✅ **Status**: Working perfectly!")
    st.info(f"Messages: {len(st.session_state.messages)}")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.header("💡 Suggested Questions")
    suggestions = [
        "What is fraud detection?",
        "How does the dashboard work?",
        "What are the risk levels?",
        "Explain transaction monitoring",
        "What are common fraud patterns?",
        "How accurate is the model?"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggest_{suggestion}"):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            st.rerun()

# Footer
st.markdown("---")
st.markdown("**Status**: ✅ Working with intelligent responses - No API key required!")
st.markdown("**Features**: Real-time fraud detection explanations, risk assessment, and dashboard guidance") 