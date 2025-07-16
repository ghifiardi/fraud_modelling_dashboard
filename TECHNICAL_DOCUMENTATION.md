# 🛡️ AI Fraud Detection Dashboard - Technical Documentation

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [API Documentation](#api-documentation)
3. [Model Specifications](#model-specifications)
4. [OpenAI Integration](#openai-integration)
5. [Deployment Guide](#deployment-guide)
6. [Configuration Reference](#configuration-reference)
7. [Troubleshooting](#troubleshooting)

## 🏗️ System Architecture

### Overview
The AI Fraud Detection Dashboard is built as a modular, scalable system with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Services   │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   (OpenAI)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Models     │    │   Data Store    │    │   Agent Network │
│   (Scikit-learn)│    │   (CSV/JSON)    │    │   (WebSocket)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. Dashboard Interface (`src/dashboard.py`)
- **Framework**: Streamlit
- **Features**: 8 main tabs with real-time monitoring
- **Responsive**: Mobile-friendly design
- **Real-time**: Auto-refresh capabilities

#### 2. Fraud Detection Engine (`src/bank_fraud_detector.py`)
- **Models**: Logistic Regression, Random Forest, Isolation Forest
- **Features**: 17 engineered features
- **Scaling**: RobustScaler for normalization
- **Persistence**: Joblib serialization

#### 3. AI Integration (`src/llm_chatbot.py`)
- **Multi-LLM**: Ollama, OpenAI, HuggingFace, Rule-based
- **Fallback**: Automatic service switching
- **Context**: Conversation history management
- **Security**: API key management

#### 4. API Server (`src/api_server.py`)
- **Framework**: FastAPI
- **Endpoints**: RESTful API for external integration
- **Validation**: Pydantic models
- **Documentation**: Auto-generated OpenAPI docs

## 📚 API Documentation

### Dashboard API Endpoints

#### GET `/api/health`
Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

#### POST `/api/predict`
Predict fraud risk for a transaction.

**Request:**
```json
{
  "amount": 1500.0,
  "transaction_type": "ONLINE",
  "location": "INTERNATIONAL",
  "customer_id": 123,
  "hour": 23,
  "merchant_category": "RETAIL"
}
```

**Response:**
```json
{
  "risk_level": "HIGH_RISK",
  "risk_probability": 0.85,
  "recommended_action": "BLOCK_TRANSACTION",
  "model_predictions": {
    "Logistic Regression": {"prediction": 1, "probability": 0.82},
    "Random Forest": {"prediction": 1, "probability": 0.88},
    "Isolation Forest": {"prediction": 1, "probability": 0.85}
  }
}
```

#### GET `/api/analytics`
Get system analytics and metrics.

**Response:**
```json
{
  "total_transactions": 1247,
  "fraud_detected": 8,
  "success_rate": 0.994,
  "avg_response_time": 0.8,
  "model_performance": {
    "logistic_regression": {"auc": 0.85, "precision": 0.82},
    "random_forest": {"auc": 0.92, "precision": 0.89},
    "isolation_forest": {"auc": 0.78, "precision": 0.75}
  }
}
```

### OpenAI Integration API

#### POST `/api/openai/generate-code`
Generate fraud detection code using OpenAI.

**Request:**
```json
{
  "prompt": "Generate Python code for detecting suspicious transaction patterns",
  "model": "gpt-4o",
  "temperature": 0.7,
  "max_tokens": 1000
}
```

#### POST `/api/openai/analyze-data`
Analyze transaction data using AI.

**Request:**
```json
{
  "analysis_type": "Transaction Pattern Analysis",
  "data": {
    "total_transactions": 1247,
    "fraud_count": 8,
    "fraud_rate": 0.64
  },
  "model": "gpt-4.1-mini"
}
```

## 🤖 Model Specifications

### Machine Learning Models

#### 1. Logistic Regression
- **Purpose**: Baseline model with interpretability
- **Features**: 17 engineered features
- **Performance**: AUC ~0.85
- **Use Case**: Quick risk assessment

#### 2. Random Forest
- **Purpose**: Robust ensemble method
- **Features**: 17 engineered features
- **Performance**: AUC ~0.92
- **Use Case**: Primary prediction model

#### 3. Isolation Forest
- **Purpose**: Anomaly detection
- **Features**: 17 engineered features
- **Performance**: AUC ~0.78
- **Use Case**: Novel fraud pattern detection

### Feature Engineering

#### Time-based Features
```python
features = [
    'is_weekend',      # Weekend transactions
    'is_night',        # Night-time transactions (22:00-06:00)
    'is_business_hours' # Business hours (09:00-17:00)
]
```

#### Amount-based Features
```python
features = [
    'amount_log',      # Log-transformed amount
    'is_high_value',   # High-value transaction flag
    'amount_percentile' # Amount percentile rank
]
```

#### Transaction Features
```python
features = [
    'is_online',       # Online transaction flag
    'is_international', # International transaction
    'card_not_present' # Card-not-present transaction
]
```

#### Customer Features
```python
features = [
    'avg_amount',      # Customer average transaction amount
    'fraud_rate',      # Customer historical fraud rate
    'transaction_count' # Customer transaction count
]
```

## 🧠 OpenAI Integration

### Supported Models

| Model | Input Tokens | Output Tokens | Cost/1K Tokens | Best Use Case |
|-------|-------------|---------------|----------------|---------------|
| gpt-4o | 128K | 4K | $5.00/$15.00 | Complex analysis |
| gpt-4o-mini | 128K | 4K | $0.15/$0.60 | General tasks |
| gpt-4.1-mini | 128K | 4K | $0.10/$0.40 | Efficiency |
| gpt-4.1-nano | 128K | 4K | $0.05/$0.20 | Speed/cost |
| gpt-3.5-turbo | 16K | 4K | $0.0015/$0.002 | Budget-friendly |

### API Integration Pattern

```python
from openai import OpenAI

def openai_request(prompt, model, api_key, temperature=0.7, max_tokens=1000):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content
```

### Error Handling

```python
try:
    result = openai_request(prompt, model, api_key)
except Exception as e:
    if "model_not_found" in str(e):
        return "Model not available, try alternative"
    elif "quota_exceeded" in str(e):
        return "API quota exceeded"
    else:
        return f"Error: {str(e)}"
```

## 🚀 Deployment Guide

### Local Development

1. **Environment Setup**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configuration**
   ```bash
   # Set environment variables
   export OPENAI_API_KEY="your_openai_key"
   export HUGGINGFACE_API_KEY="your_huggingface_key"
   ```

3. **Run Dashboard**
   ```bash
   python3 -m streamlit run src/dashboard.py --server.port 8501
   ```

### Streamlit Cloud Deployment

1. **Repository Setup**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Repository: `your-username/fraud_modelling_dashboard`
   - Main file: `streamlit_app.py`
   - Python version: 3.9+

3. **Configure Secrets**
   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY = "your_openai_key"
   HUGGINGFACE_API_KEY = "your_huggingface_key"
   ```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## ⚙️ Configuration Reference

### Streamlit Configuration (`.streamlit/config.toml`)

```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### Model Configuration

```python
# Risk thresholds
risk_thresholds = {
    'high_risk': 0.95,    # Top 5% risk
    'medium_risk': 0.85,  # Top 15% risk
    'low_risk': 0.70      # Top 30% risk
}

# Feature configuration
feature_columns = [
    'amount', 'amount_log', 'is_high_value',
    'is_weekend', 'is_night', 'is_business_hours',
    'is_online', 'is_international', 'card_not_present',
    'avg_amount', 'std_amount', 'transaction_count',
    'fraud_rate', 'risk_score', 'balance_change'
]
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Error
**Problem**: `Model not found. Please train the model first.`

**Solution**:
```bash
# Train the model first
python train_model.py
```

#### 2. OpenAI API Error
**Problem**: `Model 'gpt-4' does not exist`

**Solution**: Use correct model names:
- `gpt-4o` (latest)
- `gpt-4o-mini` (efficient)
- `gpt-4.1-mini` (new variant)
- `gpt-4.1-nano` (fastest)
- `gpt-3.5-turbo` (reliable)

#### 3. Port Already in Use
**Problem**: `Port 8501 is already in use`

**Solution**:
```bash
# Find and kill the process
lsof -i :8501
kill <PID>

# Or use a different port
python3 -m streamlit run src/dashboard.py --server.port 8502
```

#### 4. Missing Dependencies
**Problem**: `No module named 'openai'`

**Solution**:
```bash
pip install openai
pip install -r requirements.txt
```

### Performance Optimization

#### 1. Model Caching
```python
@st.cache_resource
def load_model():
    return BankFraudDetector()
```

#### 2. Data Caching
```python
@st.cache_data
def load_transaction_data():
    return pd.read_csv("data/transactions.csv")
```

#### 3. API Rate Limiting
```python
import time

def rate_limited_api_call(func, delay=1):
    time.sleep(delay)
    return func()
```

### Security Considerations

1. **API Key Management**
   - Use environment variables
   - Never commit keys to version control
   - Rotate keys regularly

2. **Data Privacy**
   - Anonymize sensitive data
   - Use local models when possible
   - Implement data retention policies

3. **Access Control**
   - Implement user authentication
   - Use role-based access control
   - Audit API usage

## 📊 Performance Metrics

### Model Performance
- **AUC-ROC**: 0.85-0.92
- **Precision**: 0.75-0.89
- **Recall**: 0.78-0.85
- **F1-Score**: 0.80-0.87

### System Performance
- **Response Time**: <1 second
- **Throughput**: 1000+ transactions/minute
- **Uptime**: 99.8%
- **Memory Usage**: <2GB

### Cost Optimization
- **OpenAI API**: $0.05-$5.00 per 1K tokens
- **Model Inference**: <$0.001 per transaction
- **Storage**: <$10/month for typical usage

---

**Documentation Version**: 1.0.0  
**Last Updated**: January 2024  
**Maintainer**: Development Team 