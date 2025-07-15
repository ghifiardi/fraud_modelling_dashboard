# 🛡️ AI Fraud Detection Monitoring Dashboard

A comprehensive monitoring and analytics dashboard for your AI fraud detection agent, featuring real-time metrics, transaction monitoring, and model management capabilities.

## 🚀 Features

### 📊 Real-time Monitoring
- **Live Metrics**: Transaction volume, fraud detection rates, success rates, and response times
- **System Health**: CPU usage, memory consumption, uptime monitoring
- **Risk Distribution**: Visual breakdown of transaction risk levels
- **Model Performance**: Real-time model accuracy and latency tracking

### 🔍 Transaction Monitoring
- **Interactive Testing**: Test individual transactions with real-time risk assessment
- **Batch Processing**: Analyze multiple transactions simultaneously
- **Transaction History**: View recent transactions with risk scores
- **Real-time Alerts**: Instant notifications for high-risk transactions

### 📈 Analytics & Insights
- **Time-based Analysis**: Hourly and daily transaction patterns
- **Customer Analysis**: Risk profiles and behavior patterns
- **Model Comparison**: Performance metrics across different models
- **Trend Analysis**: Historical performance tracking

### ⚙️ Model Management
- **Model Information**: Detailed view of loaded models and parameters
- **Performance Metrics**: Accuracy, precision, recall, and F1 scores
- **Model Health**: System resource usage and response times
- **Retraining**: Background model retraining capabilities

### 🚨 Alerts & Logs
- **Real-time Alerts**: System alerts for high-risk transactions and performance issues
- **System Logs**: Detailed transaction processing logs
- **Alert Management**: Configurable alert thresholds and notifications

## 🏗️ Architecture

The dashboard consists of three main components:

1. **FastAPI Backend** (`src/api_server.py`)
   - RESTful API endpoints for real-time data
   - Transaction processing and prediction
   - System metrics and health monitoring
   - Model management endpoints

2. **Streamlit Dashboard** (`src/dashboard.py`)
   - Interactive web-based dashboard
   - Real-time visualizations and metrics
   - Transaction testing interface
   - Model performance monitoring

3. **React Frontend** (`frontend/`)
   - Modern, responsive web interface
   - Advanced charts and visualizations
   - Real-time data updates
   - Professional UI/UX design

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for React frontend)
- npm or yarn

### Quick Start

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start All Components**
   ```bash
   python run_dashboard.py --install-deps
   ```

3. **Access Dashboards**
   - FastAPI Server: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Streamlit Dashboard: http://localhost:8501
   - React Frontend: http://localhost:3000

### Manual Installation

#### Python Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit dashboard
streamlit run src/dashboard.py --server.port 8501
```

#### React Frontend
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 📖 Usage

### Starting the Dashboard

#### Option 1: Start All Components
```bash
python run_dashboard.py
```

#### Option 2: Start Specific Components
```bash
# Start only FastAPI and Streamlit
python run_dashboard.py --components api streamlit

# Start only React frontend
python run_dashboard.py --components react
```

#### Option 3: Install Dependencies and Start
```bash
python run_dashboard.py --install-deps
```

### Using the Dashboard

#### 1. Real-time Dashboard
- View live transaction metrics and system health
- Monitor fraud detection rates and success metrics
- Track model performance in real-time

#### 2. Transaction Monitor
- Test individual transactions with the interactive form
- View transaction history and risk assessments
- Analyze batch transactions for patterns

#### 3. Analytics
- Explore time-based transaction patterns
- Analyze customer risk profiles
- Compare model performance metrics

#### 4. Model Management
- View loaded models and their parameters
- Monitor model health and performance
- Initiate model retraining

#### 5. Alerts & Logs
- Monitor system alerts and notifications
- View detailed transaction logs
- Configure alert thresholds

### API Endpoints

The FastAPI server provides the following endpoints:

#### Health & Metrics
- `GET /health` - System health check
- `GET /metrics` - Comprehensive system metrics
- `GET /models` - Model information

#### Transaction Processing
- `POST /predict` - Single transaction prediction
- `POST /predict/batch` - Batch transaction prediction
- `GET /transactions/recent` - Recent transaction history

#### Analytics
- `GET /analytics/hourly` - Hourly transaction analytics
- `GET /analytics/daily` - Daily transaction analytics

#### Monitoring
- `GET /alerts` - System alerts
- `POST /models/retrain` - Retrain models

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
REACT_APP_API_URL=http://localhost:8000

# Dashboard Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Model Configuration
MODEL_PATH=models/bank_fraud_detector.pkl
```

### Customization

#### Adding New Metrics
1. Modify `src/api_server.py` to add new endpoints
2. Update `frontend/src/services/api.js` to include new API calls
3. Add visualizations in the dashboard components

#### Custom Alerts
1. Configure alert thresholds in `src/api_server.py`
2. Add alert types in the `get_recent_alerts()` method
3. Update frontend components to display new alerts

#### Model Integration
1. Ensure your model is saved in the `models/` directory
2. Update the model path in `src/api_server.py`
3. Test the integration through the dashboard

## 📊 Dashboard Components

### Streamlit Dashboard
- **Real-time Metrics**: Live transaction and fraud detection metrics
- **Interactive Charts**: Plotly-based visualizations
- **Transaction Testing**: Form-based transaction analysis
- **Model Management**: Model information and controls

### React Frontend
- **Modern UI**: Material-UI based interface
- **Real-time Updates**: WebSocket-based live data
- **Advanced Charts**: Recharts-based visualizations
- **Responsive Design**: Mobile-friendly interface

### FastAPI Backend
- **RESTful API**: Standard HTTP endpoints
- **Real-time Processing**: Async transaction handling
- **System Monitoring**: Health checks and metrics
- **Model Management**: Model loading and prediction

## 🚨 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
lsof -i :8000
lsof -i :8501
lsof -i :3000

# Kill the process
kill -9 <PID>
```

#### Missing Dependencies
```bash
# Reinstall Python dependencies
pip install -r requirements.txt

# Reinstall React dependencies
cd frontend && npm install
```

#### Model Not Found
```bash
# Train the model first
python train_model.py

# Check model path
ls -la models/
```

#### API Connection Issues
```bash
# Check if FastAPI server is running
curl http://localhost:8000/health

# Check CORS settings in api_server.py
```

### Logs and Debugging

#### Enable Debug Logging
```python
# In src/api_server.py
logging.basicConfig(level=logging.DEBUG)
```

#### View Streamlit Logs
```bash
streamlit run src/dashboard.py --logger.level debug
```

#### Check React Console
- Open browser developer tools
- Check Console tab for errors
- Check Network tab for API calls

## 🔒 Security Considerations

### Production Deployment
- Use HTTPS for all communications
- Implement authentication and authorization
- Configure CORS properly for production domains
- Use environment variables for sensitive configuration
- Implement rate limiting for API endpoints

### Data Privacy
- Ensure transaction data is properly anonymized
- Implement data retention policies
- Use secure storage for model files
- Log only necessary information

## 📈 Performance Optimization

### Backend Optimization
- Use connection pooling for database connections
- Implement caching for frequently accessed data
- Use async processing for batch operations
- Monitor memory usage and implement garbage collection

### Frontend Optimization
- Implement lazy loading for components
- Use React.memo for expensive components
- Optimize chart rendering with proper data structures
- Implement proper error boundaries

## 🤝 Contributing

### Adding New Features
1. Create a feature branch
2. Implement the feature with tests
3. Update documentation
4. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use ESLint for JavaScript/React code
- Add type hints for Python functions
- Write comprehensive docstrings

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation at http://localhost:8000/docs
- Open an issue in the project repository

---

**Happy Monitoring! 🛡️📊** 