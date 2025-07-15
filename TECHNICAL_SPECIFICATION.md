# Fraud Detection System - Technical Specification

## System Architecture

### Overview
The fraud detection system is built as a modular, scalable architecture with the following components:

1. **Core ML Engine** (BankFraudDetector)
2. **Web Dashboard** (Streamlit)
3. **API Server** (FastAPI)
4. **Multi-Agent Pipeline** (LangGraph)

### Component Interaction
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   Multi-Agent   │
│   Dashboard     │◄──►│   Server        │◄──►│   Pipeline      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Bank Fraud    │
                    │   Detector      │
                    │   (Core ML)     │
                    └─────────────────┘
```

## Data Flow

### 1. Transaction Input
```python
transaction_data = {
    "transaction_id": "TXN_001",
    "amount": 150.00,
    "customer_id": "CUST_001",
    "transaction_type": "ONLINE",
    "hour": 14,
    "day_of_week": 2,
    "location": "DOMESTIC",
    "device_type": "MOBILE",
    "card_present": False
}
```

### 2. Feature Engineering Pipeline
```python
def engineer_bank_features(df):
    # Time-based features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    # Amount-based features
    df['amount_log'] = np.log1p(df['amount'])
    df['is_high_value'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    
    # Transaction features
    df['is_online'] = (df['transaction_type'] == 'ONLINE').astype(int)
    df['is_international'] = (df['location'] == 'INTERNATIONAL').astype(int)
    df['card_not_present'] = (~df['card_present']).astype(int)
    
    return df
```

### 3. Model Prediction
```python
def predict_transaction_risk(transaction_data):
    # Feature preparation
    features = prepare_features(transaction_data)
    
    # Model ensemble prediction
    predictions = {}
    for model_name, model in self.models.items():
        pred = model.predict(features)
        prob = model.predict_proba(features)[0, 1]
        predictions[model_name] = {'prediction': pred, 'probability': prob}
    
    # Risk level determination
    risk_level = determine_risk_level(predictions)
    
    return {
        'risk_level': risk_level,
        'risk_probability': max_prob,
        'model_predictions': predictions,
        'recommended_action': get_action(risk_level)
    }
```

## Machine Learning Models

### 1. Logistic Regression
- **Purpose**: Baseline model with interpretability
- **Advantages**: Fast training, interpretable coefficients
- **Use Case**: Initial fraud screening

### 2. Random Forest
- **Purpose**: Robust ensemble method
- **Advantages**: Handles non-linear relationships, feature importance
- **Use Case**: Primary fraud detection model

### 3. Isolation Forest
- **Purpose**: Anomaly detection
- **Advantages**: Detects outliers without labeled fraud data
- **Use Case**: Unsupervised fraud detection

### Model Training Process
```python
def train_bank_models(df, test_size=0.2):
    # Feature preparation
    X = df[feature_columns]
    y = df[fraud_column]
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Isolation Forest': IsolationForest(random_state=42, contamination=y_train.mean())
    }
    
    # Training and evaluation
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        auc_score = roc_auc_score(y_test, y_pred)
        
    return results, X_test_scaled, y_test
```

## Feature Engineering Specification

### Time-Based Features
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `is_weekend` | Weekend transaction flag | `day_of_week in [5, 6]` |
| `is_night` | Night-time transaction flag | `hour >= 22 or hour <= 6` |
| `is_business_hours` | Business hours flag | `9 <= hour <= 17` |

### Amount-Based Features
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `amount_log` | Log-transformed amount | `log(1 + amount)` |
| `is_high_value` | High-value transaction flag | `amount > 95th percentile` |
| `amount_percentile` | Amount percentile rank | `rank(amount) / total_count` |

### Transaction Features
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `is_online` | Online transaction flag | `transaction_type == 'ONLINE'` |
| `is_atm` | ATM transaction flag | `transaction_type == 'ATM'` |
| `is_international` | International transaction flag | `location == 'INTERNATIONAL'` |
| `card_not_present` | Card-not-present flag | `not card_present` |

### Customer Behavior Features
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `avg_amount` | Customer average transaction amount | `groupby(customer_id)['amount'].mean()` |
| `std_amount` | Customer transaction amount std | `groupby(customer_id)['amount'].std()` |
| `transaction_count` | Customer transaction count | `groupby(customer_id)['amount'].count()` |
| `fraud_rate` | Customer historical fraud rate | `groupby(customer_id)['is_fraud'].mean()` |

## Risk Assessment Framework

### Risk Levels
1. **SAFE** (0-30th percentile)
   - Action: Allow transaction
   - Monitoring: Standard

2. **LOW_RISK** (30-70th percentile)
   - Action: Monitor closely
   - Monitoring: Enhanced

3. **MEDIUM_RISK** (70-85th percentile)
   - Action: Require additional verification
   - Monitoring: High

4. **HIGH_RISK** (85-100th percentile)
   - Action: Block transaction
   - Monitoring: Immediate

### Risk Score Calculation
```python
def calculate_risk_score(transaction):
    risk_score = (
        transaction['is_high_value'] * 2 +
        transaction['is_international'] * 3 +
        transaction['card_not_present'] * 2 +
        transaction['is_night'] * 1 +
        transaction['previous_fraud_flag'] * 5
    )
    return risk_score
```

## API Specification

### Endpoints

#### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 3,
  "last_training": "2024-12-19T10:30:00Z",
  "version": "1.0.0"
}
```

#### 2. Predict Transaction
```http
POST /predict
Content-Type: application/json
```
**Request Body:**
```json
{
  "transaction_id": "TXN_001",
  "amount": 150.00,
  "customer_id": "CUST_001",
  "transaction_type": "ONLINE",
  "hour": 14,
  "day_of_week": 2,
  "location": "DOMESTIC",
  "device_type": "MOBILE",
  "card_present": false
}
```
**Response:**
```json
{
  "transaction_id": "TXN_001",
  "risk_level": "LOW_RISK",
  "risk_probability": 0.25,
  "recommended_action": "MONITOR_CLOSELY",
  "model_predictions": {
    "Logistic Regression": {"prediction": 0, "probability": 0.20},
    "Random Forest": {"prediction": 0, "probability": 0.25},
    "Isolation Forest": {"prediction": 0, "probability": 0.30}
  }
}
```

#### 3. Model Information
```http
GET /model-info
```
**Response:**
```json
{
  "models": ["Logistic Regression", "Random Forest", "Isolation Forest"],
  "performance_metrics": {
    "auc_scores": {
      "Logistic Regression": 0.85,
      "Random Forest": 0.92,
      "Isolation Forest": 0.78
    }
  },
  "feature_count": 17,
  "training_date": "2024-12-19T10:30:00Z"
}
```

## Dashboard Specification

### Layout Structure
```
┌─────────────────────────────────────────────────────────────┐
│                    Header (Title + Status)                  │
├─────────────────────────────────────────────────────────────┤
│  Metrics Row: AUC, Precision, Recall, F1-Score             │
├─────────────────────────────────────────────────────────────┤
│  Main Content Area                                          │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   Transaction   │  │   Performance   │                  │
│  │   Feed          │  │   Charts        │                  │
│  └─────────────────┘  └─────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│  Alerts & Logs Section                                      │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   Risk Alerts   │  │   Transaction   │                  │
│  │   (Cards)       │  │   Logs          │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Metrics Dashboard
- **AUC Score**: Model discrimination ability
- **Precision**: Accuracy of positive predictions
- **Recall**: Sensitivity to fraud detection
- **F1-Score**: Harmonic mean of precision and recall

#### 2. Transaction Feed
- Real-time transaction display
- Risk level indicators
- Transaction details on hover

#### 3. Performance Charts
- ROC curves for all models
- Confusion matrices
- Feature importance plots

#### 4. Alert System
- Color-coded risk alerts
- High risk: Red cards
- Medium risk: Orange cards
- Low risk: Yellow cards

#### 5. Transaction Logs
- Detailed transaction history
- Timestamp and risk information
- Search and filter capabilities

## Performance Requirements

### Response Times
- **Dashboard Load**: < 3 seconds
- **Transaction Prediction**: < 500ms
- **API Response**: < 200ms
- **Model Training**: < 5 minutes (for 5K transactions)

### Scalability
- **Concurrent Users**: 10+ dashboard users
- **Transaction Throughput**: 100+ transactions/minute
- **Data Volume**: 10K+ transactions in memory

### Accuracy Targets
- **AUC Score**: > 0.85
- **Precision**: > 0.80 (at 0.90 recall)
- **Recall**: > 0.90 (for fraud detection)
- **False Positive Rate**: < 0.10

## Security Requirements

### Data Protection
- No PII storage in models
- Transaction data anonymization
- Secure model serialization

### Access Control
- Local deployment for sensitive data
- Network-level security for shared access
- API authentication for production

### Input Validation
- Transaction data validation
- Feature range checking
- Error handling and logging

## Deployment Specifications

### Local Development
```bash
# Requirements
Python 3.8+
8GB RAM
2GB disk space

# Dependencies
pip install -r requirements.txt

# Startup
python3 -m streamlit run src/dashboard.py --server.port 8501
```

### Production Deployment
```bash
# Docker
docker build -t fraud-detection-dashboard .
docker run -p 8501:8501 fraud-detection-dashboard

# Streamlit Cloud
# Connect GitHub repository
# Deploy with main file: src/dashboard.py
```

### Monitoring
- Application health checks
- Model performance monitoring
- Error logging and alerting
- Resource usage tracking

---

**Document Version**: 1.0.0
**Last Updated**: December 2024
**Author**: AI Assistant 