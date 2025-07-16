# 🛡️ AI Fraud Detection Dashboard

A comprehensive fraud detection system with real-time monitoring, machine learning models, multi-agent intelligence network, and advanced AI capabilities powered by OpenAI.

## 🌟 Features

### **Core Fraud Detection**
- **Real-time Fraud Detection**: Monitor transactions in real-time with ML-powered risk scoring
- **Multi-Model Ensemble**: Logistic Regression, Random Forest, and Isolation Forest
- **Analytics Dashboard**: Comprehensive charts and performance metrics
- **Analyst Review System**: Manual review and feedback collection
- **Model Management**: Real-time model performance monitoring

### **AI-Powered Capabilities**
- **OpenAI Playground Integration**: 5 AI-powered features with latest GPT models
- **Multi-LLM Chatbot**: Support for Ollama, OpenAI, HuggingFace, and rule-based responses
- **Code Generation**: Generate fraud detection code with AI
- **Data Analysis**: AI-powered transaction pattern analysis
- **Report Generation**: Automated fraud analysis reports
- **Model Explanation**: Explain why transactions are flagged

### **Advanced Features**
- **Fraud Intelligence Network**: Connect with other agents and systems
- **Multi-Agent Communication**: Real-time intelligence sharing
- **Indonesian Banking Integration**: BI-FAST and local bank consortium support
- **Global Fraud Networks**: SWIFT, Visa, Mastercard integration ready

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
- Risk distribution analysis

### 2. Transaction Monitor
- Individual transaction analysis
- Risk scoring and recommendations
- Recent transaction history
- Custom transaction testing

### 3. Analytics
- Model performance metrics
- Transaction patterns
- Risk distribution analysis
- Customer behavior insights

### 4. Model Management
- Model status and health
- Performance monitoring
- Configuration settings
- Model comparison

### 5. Alerts & Logs
- Real-time alerts
- System logs
- Risk notifications
- Alert history

### 6. Analyst Review
- Manual transaction review
- Feedback collection
- Review history
- Label management

### 7. Fraud Intelligence Network
- Multi-agent communication
- Real-time intelligence sharing
- Network configuration
- Agent status monitoring

### 8. OpenAI Playground
- **Code Generation**: Generate fraud detection code
- **Data Analysis**: AI-powered pattern analysis
- **Report Generation**: Automated reports
- **Model Explanation**: Explain predictions
- **Custom Prompts**: Interactive AI assistance

## 🤖 AI Chatbot Assistant

The dashboard includes an intelligent chatbot that supports multiple LLM providers:

- **Ollama** (Local): For privacy-focused deployments
- **OpenAI**: For advanced reasoning capabilities
- **HuggingFace**: For open-source model access
- **Rule-based**: Fallback responses for reliability

### Available OpenAI Models
| Model | Description | Best For |
|-------|-------------|----------|
| **gpt-4o** | Latest and most capable | Complex tasks, best quality |
| **gpt-4o-mini** | Fast and efficient | Good balance of speed/capability |
| **gpt-4.1-mini** | New GPT-4.1 variant | Optimized for efficiency |
| **gpt-4.1-nano** | Smallest GPT-4.1 model | Fastest, most cost-effective |
| **gpt-3.5-turbo** | Reliable and cost-effective | Most tasks, good value |

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

### OpenAI Integration
- **API Key**: Enter your OpenAI API key in the Playground tab
- **Model Selection**: Choose from 5 different models
- **Temperature Control**: Adjust creativity (0.0-2.0)
- **Token Limits**: Control response length (100-4000 tokens)

## 📈 Performance

- **Real-time Processing**: Sub-second transaction analysis
- **High Accuracy**: Multi-model ensemble approach
- **Scalable**: Designed for production banking environments
- **Low False Positives**: Optimized risk thresholds
- **AI Integration**: Seamless OpenAI API integration

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python, FastAPI
- **ML Models**: Scikit-learn, XGBoost, LightGBM
- **Visualization**: Plotly, Matplotlib
- **LLM Integration**: OpenAI, HuggingFace, Ollama
- **Data Processing**: Pandas, NumPy
- **AI Services**: OpenAI GPT-4.1, GPT-4o models

## 🌐 Multi-Agent Intelligence Network

### Connected Agents
- **Jakarta Bank Consortium Agent**: BI-FAST fraud patterns
- **Singapore Regional Agent**: ASEAN fraud trends
- **Global AML Network Agent**: International money laundering

### Intelligence Sharing
- Real-time fraud pattern sharing
- Cross-border threat intelligence
- Automated alert distribution
- Network health monitoring

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
