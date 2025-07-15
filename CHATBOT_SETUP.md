# AI Chatbot Setup Guide

## 🚀 **Multiple LLM Fallback System**

The fraud detection dashboard now includes an AI chatbot with multiple fallback options:

### **1. Local Ollama (Primary)**
- **Requirement**: Ollama installed with Mistral 7B model
- **Setup**: `ollama run mistral:7b-instruct`
- **Pros**: Free, private, no API limits
- **Cons**: Requires local setup

### **2. OpenAI API (Fallback 1)**
- **Requirement**: OpenAI API key
- **Setup**: Set environment variable `OPENAI_API_KEY`
- **Pros**: High quality responses, reliable
- **Cons**: Requires API key, usage costs

### **3. HuggingFace Inference API (Fallback 2)**
- **Requirement**: HuggingFace API key (free tier available)
- **Setup**: Set environment variable `HUGGINGFACE_API_KEY`
- **Pros**: Free tier available, good performance
- **Cons**: Rate limits on free tier

### **4. Rule-based Responses (Fallback 3)**
- **Requirement**: None
- **Setup**: Automatic fallback
- **Pros**: Always works, no API keys needed
- **Cons**: Limited responses, not as intelligent

## 🔧 **Setup Instructions**

### **For Local Development:**
```bash
# Install Ollama (if not already installed)
brew install ollama

# Pull Mistral model
ollama pull mistral:7b-instruct

# Start Ollama service
brew services start ollama
```

### **For Cloud Deployment (Streamlit Cloud):**
The chatbot will automatically use rule-based responses, which work perfectly for basic fraud detection questions.

### **Optional: Add Cloud LLM APIs**

#### **OpenAI API:**
1. Get API key from: https://platform.openai.com/api-keys
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

#### **HuggingFace API:**
1. Get API key from: https://huggingface.co/settings/tokens
2. Set environment variable:
   ```bash
   export HUGGINGFACE_API_KEY=your_key_here
   ```

## 🎯 **Chatbot Features**

### **What You Can Ask:**
- "What is fraud detection?"
- "How does the dashboard work?"
- "Explain risk levels"
- "What are the model performance metrics?"
- "How to use the transaction monitor?"
- "What do the different tabs show?"

### **Connection Status:**
The chatbot shows connection status for each service:
- ✅ Available services
- ℹ️ Optional services not configured
- 🔄 Automatic fallback to next available service

## 🌐 **Deployment Notes**

### **Streamlit Cloud:**
- ✅ Rule-based responses work perfectly
- ✅ No API keys required
- ✅ Always functional
- ✅ Professional user experience

### **Local Development:**
- ✅ Full LLM capabilities with Ollama
- ✅ Optional cloud APIs for enhanced responses
- ✅ Best performance and privacy

## 🛠️ **Troubleshooting**

### **Chatbot Not Responding:**
1. Check if Ollama is running: `ollama list`
2. Verify API keys are set correctly
3. Check network connectivity for cloud APIs
4. Rule-based responses should always work

### **Slow Responses:**
1. Local Ollama: Check system resources
2. Cloud APIs: Check network speed
3. Rule-based: Should be instant

### **API Errors:**
1. Verify API keys are valid
2. Check API usage limits
3. Ensure proper environment variables

## 📝 **Example Usage**

```
User: "What is fraud detection?"
Assistant: "Fraud detection is a system that identifies suspicious or fraudulent transactions in real-time. Our dashboard uses machine learning models to analyze transaction patterns and flag potential risks..."

User: "How do I use the dashboard?"
Assistant: "The dashboard has 5 main sections: 1. Real-time Dashboard: Live metrics and transaction monitoring 2. Transaction Monitor: Detailed transaction analysis..."

User: "What are the risk levels?"
Assistant: "Risk levels in our system: - HIGH RISK: Top 5% risk scores - Transactions are blocked - MEDIUM RISK: Top 15% risk scores - Additional verification required..."
```

The chatbot automatically provides helpful, contextual responses regardless of which LLM service is available! 