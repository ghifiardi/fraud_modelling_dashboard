# 🛡️ AI Fraud Detection Dashboard

A comprehensive fraud detection system with real-time monitoring, machine learning models, and multi-agent intelligence network.

## 🌟 Features

- **Real-time Fraud Detection**: Monitor transactions in real-time with ML-powered risk scoring
- **Multi-Model Ensemble**: Logistic Regression, Random Forest, and Isolation Forest
- **Analytics Dashboard**: Comprehensive charts and performance metrics
- **Analyst Review System**: Manual review and feedback collection
- **AI Chatbot Assistant**: Multi-LLM support (Ollama, OpenAI, HuggingFace)
- **Fraud Intelligence Network**: Connect with other agents and systems
- **Model Management**: Real-time model performance monitoring

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/ghifiardi/fraud_modelling_dashboard.git
   cd fraud_modelling_dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   python3 -m streamlit run src/dashboard.py --server.port 8501
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### 🎯 Streamlit Cloud Deployment

This dashboard is ready for deployment on Streamlit Cloud!

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
3. **Connect your GitHub account**
4. **Deploy the app**:
   - Repository: `your-username/fraud_modelling_dashboard`
   - Main file path: `streamlit_app.py`
   - Python version: 3.9+

## 📊 Dashboard Sections

### 1. Real-time Dashboard
- Live transaction monitoring
- Key performance metrics
- Real-time charts and visualizations

### 2. Transaction Monitor
- Individual transaction analysis
- Risk scoring and recommendations
- Recent transaction history

### 3. Analytics
- Model performance metrics
- Transaction patterns
- Risk distribution analysis

### 4. Model Management
- Model status and health
- Performance monitoring
- Configuration settings

### 5. Alerts & Logs
- Real-time alerts
- System logs
- Risk notifications

### 6. Analyst Review
- Manual transaction review
- Feedback collection
- Review history

### 7. Fraud Intelligence Network
- Multi-agent communication
- Real-time intelligence sharing
- Network configuration

## 🤖 AI Chatbot Assistant

The dashboard includes an intelligent chatbot that supports multiple LLM providers:

- **Ollama** (Local): For privacy-focused deployments
- **OpenAI**: For advanced reasoning capabilities
- **HuggingFace**: For open-source model access
- **Rule-based**: Fallback responses for reliability

## 🔧 Configuration

### Environment Variables
```bash
# Optional: For enhanced chatbot functionality
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_API_KEY=your_huggingface_key
```

### Model Configuration
- Models are automatically loaded from `models/bank_fraud_detector.pkl`
- Risk thresholds are configurable in the dashboard
- Real-time settings can be adjusted in the sidebar

## 📈 Performance

- **Real-time Processing**: Sub-second transaction analysis
- **High Accuracy**: Multi-model ensemble approach
- **Scalable**: Designed for production banking environments
- **Low False Positives**: Optimized risk thresholds

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python, FastAPI
- **ML Models**: Scikit-learn, XGBoost, LightGBM
- **Visualization**: Plotly, Matplotlib
- **LLM Integration**: OpenAI, HuggingFace, Ollama
- **Data Processing**: Pandas, NumPy

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for secure financial transactions**
