# AI-Driven Fraud Detection System - Complete Documentation

## 🎯 Project Overview

This comprehensive fraud detection system combines machine learning algorithms with a modern web dashboard for real-time fraud monitoring. The system is designed for bank-level constraints with small datasets and domain-specific features.

## 📊 System Architecture

### Core Components

1. **Bank Fraud Detector** (`src/bank_fraud_detector.py`)
   - Main fraud detection engine
   - Multiple ML algorithms (Logistic Regression, Random Forest, Isolation Forest)
   - Real-time transaction scoring
   - Customer risk profiling

2. **Streamlit Dashboard** (`src/dashboard.py`)
   - Real-time monitoring interface
   - Transaction analysis and visualization
   - Alert system with color-coded notifications
   - Performance metrics display

3. **FastAPI Server** (`src/api_server.py`)
   - RESTful API endpoints
   - Real-time prediction API
   - Model management endpoints
   - Health monitoring

4. **Multi-Agent Pipeline** (`src/multi_agent_pipeline.py`)
   - Advanced fraud detection using multiple AI agents
   - LangGraph-based workflow
   - Enriched transaction analysis

## 🚀 Quick Start Guide

### 1. Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python3 -m streamlit run src/dashboard.py --server.port 8501

# Run the API server (optional)
python3 -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Dashboard Access
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.68.106:8501 (for sharing on local network)

### 3. API Endpoints
- **Health Check**: http://localhost:8000/health
- **Predict Transaction**: http://localhost:8000/predict
- **Model Info**: http://localhost:8000/model-info

## 📈 Dashboard Features

### Real-Time Monitoring
- **Transaction Feed**: Live transaction monitoring
- **Risk Scoring**: Real-time fraud probability calculation
- **Alert System**: Color-coded alerts (High/Medium/Low Risk)
- **Performance Metrics**: Model accuracy and performance indicators

### Analytics & Visualization
- **Transaction Distribution**: Amount and time-based analysis
- **Fraud Patterns**: Historical fraud trend analysis
- **Customer Profiles**: Risk categorization and behavior patterns
- **Model Performance**: ROC curves, confusion matrices, feature importance

### Alert System
- **High Risk Alerts**: Red cards for immediate attention
- **Medium Risk Alerts**: Orange cards for monitoring
- **Low Risk Alerts**: Yellow cards for awareness
- **Log Display**: Detailed transaction logs with timestamps

## 🔧 Technical Implementation

### Bank Fraud Detector Class

```python
class BankFraudDetector:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.customer_profiles = {}
        self.risk_thresholds = {}
```

**Key Methods:**
- `load_bank_data()`: Load and validate transaction data
- `engineer_bank_features()`: Create domain-specific features
- `train_bank_models()`: Train multiple ML algorithms
- `predict_transaction_risk()`: Real-time risk scoring
- `create_customer_profiles()`: Customer risk categorization

### Feature Engineering

**Time-based Features:**
- Weekend transactions
- Night-time transactions
- Business hours analysis

**Amount-based Features:**
- Log-transformed amounts
- High-value transaction flags
- Amount percentiles

**Transaction Features:**
- Online vs ATM transactions
- International transaction flags
- Card-present vs card-not-present

**Customer Behavior Features:**
- Average transaction amounts
- Transaction frequency
- Historical fraud rates
- Risk scores

### Model Training

**Supported Algorithms:**
1. **Logistic Regression**: Baseline model with interpretability
2. **Random Forest**: Robust ensemble method
3. **Isolation Forest**: Anomaly detection approach

**Training Process:**
- Feature scaling with RobustScaler
- Missing value imputation
- Stratified train-test split
- Cross-validation
- Hyperparameter optimization

## 📊 Performance Metrics

### Model Evaluation
- **AUC-ROC Score**: Area under the ROC curve
- **Precision**: Accuracy of positive predictions
- **Recall**: Sensitivity to fraud detection
- **F1-Score**: Harmonic mean of precision and recall

### Risk Thresholds
- **High Risk**: Top 5% of risk scores
- **Medium Risk**: Top 15% of risk scores
- **Low Risk**: Top 30% of risk scores
- **Safe**: Below 30th percentile

## 🌐 Deployment Options

### 1. Local Development
```bash
# Start dashboard only
python3 -m streamlit run src/dashboard.py --server.port 8501

# Start complete system
python3 start_dashboard.py
```

### 2. Network Sharing
```bash
# Make dashboard accessible on local network
python3 -m streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### 3. Streamlit Cloud Deployment
1. Push code to GitHub (public repository)
2. Connect to Streamlit Cloud
3. Deploy with main file: `src/dashboard.py`
4. Get public URL for sharing

### 4. Docker Deployment
```bash
# Build and run with Docker
docker build -t fraud-detection-dashboard .
docker run -p 8501:8501 fraud-detection-dashboard
```

## 🔍 API Documentation

### Endpoints

#### Health Check
```http
GET /health
```
Returns system health status and model information.

#### Predict Transaction
```http
POST /predict
Content-Type: application/json

{
  "transaction_id": "12345",
  "amount": 150.00,
  "customer_id": "CUST001",
  "transaction_type": "ONLINE",
  "hour": 14,
  "day_of_week": 2
}
```

#### Model Information
```http
GET /model-info
```
Returns detailed model performance metrics and configuration.

## 📁 Project Structure

```
fraud_modelling_project/
├── src/
│   ├── bank_fraud_detector.py    # Main fraud detection engine
│   ├── dashboard.py              # Streamlit dashboard
│   ├── api_server.py             # FastAPI server
│   ├── multi_agent_pipeline.py   # Advanced AI pipeline
│   ├── feature_engineering.py    # Feature creation utilities
│   └── data_downloader.py        # Dataset management
├── models/
│   └── bank_fraud_detector.pkl   # Trained model
├── data/
│   ├── raw/                      # Original datasets
│   └── processed/                # Processed datasets
├── notebooks/
│   └── fraud_detection_workflow.ipynb
├── requirements.txt              # Python dependencies
├── start_dashboard.py           # System startup script
└── README.md                    # Main documentation
```

## 🛠️ Customization

### Adding New Features
1. Extend `engineer_bank_features()` method
2. Update feature selection in `train_bank_models()`
3. Modify dashboard visualizations

### Adding New Models
1. Import new algorithm in `train_bank_models()`
2. Add to models dictionary
3. Update prediction logic in `predict_transaction_risk()`

### Customizing Risk Thresholds
```python
# Modify in set_risk_thresholds() method
thresholds = {
    'high_risk': np.percentile(probabilities, 95),
    'medium_risk': np.percentile(probabilities, 85),
    'low_risk': np.percentile(probabilities, 70)
}
```

## 🔒 Security Considerations

### Data Privacy
- No sensitive customer data stored
- Transaction IDs are anonymized
- Model doesn't require PII

### Access Control
- Local deployment for sensitive environments
- Network-level security for shared access
- API authentication for production use

### Model Security
- Model files are serialized securely
- Input validation on all endpoints
- Error handling prevents information leakage

## 📈 Future Enhancements

### Planned Features
1. **Real-time Data Integration**: Connect to live transaction feeds
2. **Advanced Analytics**: Deep learning models and neural networks
3. **Multi-language Support**: Internationalization for global deployment
4. **Mobile Dashboard**: Responsive design for mobile devices
5. **Advanced Reporting**: Automated report generation

### Scalability Improvements
1. **Database Integration**: PostgreSQL/MongoDB for data persistence
2. **Message Queues**: Redis/RabbitMQ for high-throughput processing
3. **Microservices**: Containerized deployment with Kubernetes
4. **Load Balancing**: Multiple dashboard instances

## 🆘 Troubleshooting

### Common Issues

#### Dashboard Not Loading
```bash
# Check if port is available
lsof -i :8501

# Try different port
python3 -m streamlit run src/dashboard.py --server.port 8502
```

#### Model Loading Errors
```bash
# Rebuild model
python3 -c "from src.bank_fraud_detector import BankFraudDetector; detector = BankFraudDetector(); detector.save_bank_model('models/bank_fraud_detector.pkl')"
```

#### Missing Dependencies
```bash
# Install requirements
pip install -r requirements.txt

# For specific packages
pip install streamlit fastapi uvicorn
```

### Performance Optimization
1. **Reduce Data Size**: Use smaller sample datasets for testing
2. **Optimize Features**: Select only essential features
3. **Caching**: Enable Streamlit caching for repeated computations
4. **Background Processing**: Use async processing for heavy computations

## 📞 Support

For technical support or questions:
1. Check the troubleshooting section above
2. Review the code documentation in source files
3. Examine the Jupyter notebook for detailed examples
4. Check GitHub issues for known problems

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Author**: AI Assistant
**License**: MIT 