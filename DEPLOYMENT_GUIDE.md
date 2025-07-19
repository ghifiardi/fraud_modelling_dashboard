# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

Before deploying to Streamlit Cloud, ensure you have:

1. **GitHub Account**: Your code must be in a GitHub repository
2. **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **API Keys**: Configure your OpenAI API key in Streamlit Cloud secrets

## 🔧 Deployment Steps

### Step 1: Prepare Your Repository

Ensure your repository structure looks like this:
```
fraud_modelling_project/
├── streamlit_app.py          # Main entry point
├── requirements.txt          # Python dependencies
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml         # API keys (local only)
├── src/
│   ├── dashboard.py         # Main dashboard
│   ├── bank_fraud_detector.py
│   ├── streaming_fraud_detector.py
│   └── llm_chatbot.py
├── models/
│   └── bank_fraud_detector.pkl
└── README.md
```

### Step 2: Push to GitHub

```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit for Streamlit Cloud deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fraud_modelling_project.git
git push -u origin main
```

### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
2. **Sign in**: Use your GitHub account
3. **New App**: Click "New app"
4. **Configure App**:
   - **Repository**: Select your fraud detection repository
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose a custom URL (optional)

### Step 4: Configure Secrets

In Streamlit Cloud dashboard:

1. **Go to App Settings** → **Secrets**
2. **Add your OpenAI API key**:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

### Step 5: Deploy

Click **"Deploy!"** and wait for the build to complete.

## 🌐 Access Your App

Your app will be available at:
```
https://YOUR_APP_NAME-YOUR_USERNAME.streamlit.app
```

## 🔍 Features Available in Cloud

### ✅ Working Features
- **Real-time Dashboard**: Live fraud detection metrics
- **Analytics**: Indonesian geographic distribution
- **Model Management**: Fraud detection model status
- **Streaming System**: Apache Kafka + Spark simulation
- **AI Chatbot**: OpenAI-powered assistance
- **FraudLabs Pro**: Real transaction screening

### 📊 Dashboard Tabs
1. **📊 Real-time Dashboard**: Live metrics and monitoring
2. **🚀 Streaming System**: Real-time fraud detection simulation
3. **🔍 Transaction Monitor**: Detailed transaction analysis
4. **📈 Analytics**: Charts and performance metrics
5. **⚙️ Model Management**: Model status and configuration
6. **🚨 Alerts & Logs**: Risk alerts and transaction logs
7. **📝 Analyst Review**: Manual transaction review
8. **🌐 Fraud Intelligence Network**: External integrations
9. **🤖 OpenAI Playground**: AI-powered analysis

## 🛠️ Troubleshooting

### Common Issues

#### 1. **Import Errors**
- Ensure all dependencies are in `requirements.txt`
- Check that `src/` directory is properly structured

#### 2. **API Key Issues**
- Verify OpenAI API key is set in Streamlit Cloud secrets
- Check API key format and validity

#### 3. **Model Loading Errors**
- Ensure `models/bank_fraud_detector.pkl` exists
- Model will auto-train if not found

#### 4. **Performance Issues**
- Reduce TPS in streaming system (default: 75 TPS)
- Monitor resource usage in Streamlit Cloud dashboard

### Debug Commands

```bash
# Check local deployment
streamlit run streamlit_app.py

# Test requirements
pip install -r requirements.txt

# Verify model file
ls -la models/
```

## 📈 Monitoring

### Streamlit Cloud Metrics
- **App Performance**: Monitor in Streamlit Cloud dashboard
- **Error Logs**: Check deployment logs for issues
- **Resource Usage**: Monitor CPU and memory usage

### Performance Optimization
- **Caching**: Use `@st.cache_data` for expensive operations
- **Lazy Loading**: Load models only when needed
- **Batch Processing**: Process transactions in batches

## 🔄 Updates

### Updating Your App
1. **Make Changes**: Update your local code
2. **Commit & Push**: Push to GitHub
3. **Auto-Deploy**: Streamlit Cloud automatically redeploys

### Version Control
```bash
# Update and deploy
git add .
git commit -m "Update fraud detection features"
git push origin main
```

## 🎯 Best Practices

### Code Organization
- Keep main logic in `src/` directory
- Use `streamlit_app.py` as entry point only
- Separate configuration from code

### Security
- Never commit API keys to GitHub
- Use Streamlit Cloud secrets for sensitive data
- Validate user inputs

### Performance
- Cache expensive operations
- Use efficient data structures
- Monitor resource usage

## 📞 Support

### Getting Help
- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **GitHub Issues**: Report bugs in your repository
- **Community**: [discuss.streamlit.io](https://discuss.streamlit.io)

### Useful Links
- [Streamlit Cloud](https://share.streamlit.io)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Python Requirements](https://pip.pypa.io/en/stable/reference/requirements-file-format/)

## 🎉 Success!

Once deployed, your fraud detection dashboard will be:
- **Accessible worldwide** via web browser
- **Real-time** with live fraud detection
- **Scalable** with Streamlit Cloud infrastructure
- **Secure** with proper API key management

Your Indonesian fraud detection system is now live! 🇮🇩🛡️ 