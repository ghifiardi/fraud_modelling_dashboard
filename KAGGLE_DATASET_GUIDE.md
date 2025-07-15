# Kaggle Dataset Guide for Fraud Detection

This guide will help you download and use real fraud detection datasets from Kaggle to test the fraud modeling code.

## 🎯 Available Datasets

### 1. Credit Card Fraud Detection (Recommended)
- **URL**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size**: ~150 MB
- **Features**: 30 (28 anonymized + Amount + Time)
- **Fraud Rate**: ~0.17%
- **Transactions**: ~284K
- **Best for**: Credit card transaction fraud

### 2. IEEE-CIS Fraud Detection
- **URL**: https://www.kaggle.com/c/ieee-fraud-detection/data
- **Size**: ~1.5 GB
- **Features**: 400+
- **Fraud Rate**: ~3.5%
- **Best for**: Large-scale fraud detection

### 3. Synthetic Financial Dataset (PaySim)
- **URL**: https://www.kaggle.com/datasets/ealaxi/paysim1
- **Size**: ~200 MB
- **Features**: 11
- **Fraud Rate**: ~0.6%
- **Best for**: Mobile money fraud

## 📥 Download Methods

### Method 1: Kaggle API (Recommended)

1. **Create Kaggle Account**
   - Go to https://www.kaggle.com
   - Sign up for a free account

2. **Get API Credentials**
   - Go to https://www.kaggle.com/settings/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - Download `kaggle.json`

3. **Setup API Credentials**
   ```bash
   # Create kaggle directory
   mkdir -p ~/.kaggle
   
   # Copy kaggle.json to the directory
   cp path/to/downloaded/kaggle.json ~/.kaggle/
   
   # Set correct permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Download Dataset**
   ```bash
   # Download Credit Card Fraud dataset
   kaggle datasets download mlg-ulb/creditcardfraud -p data/raw --unzip
   
   # Or use the provided script
   python3 download_kaggle_dataset.py
   ```

### Method 2: Manual Download

1. **Visit Dataset Page**
   - Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Click "Download" button

2. **Extract and Place**
   ```bash
   # Extract the downloaded zip file
   unzip creditcardfraud.zip
   
   # Move to project directory
   mv creditcard.csv data/raw/
   ```

3. **Verify Download**
   ```bash
   # Check file exists
   ls -la data/raw/creditcard.csv
   
   # Verify with Python
   python3 -c "import pandas as pd; df=pd.read_csv('data/raw/creditcard.csv'); print(f'Shape: {df.shape}, Fraud rate: {df.Class.mean():.4f}')"
   ```

## 🧪 Testing with Real Data

### Quick Test
```bash
# Run the comprehensive test
python3 test_fraud_modeling.py
```

### Full Pipeline
```bash
# 1. Data exploration
python3 data_exploration.py

# 2. Model training
python3 train_model.py

# 3. Interactive analysis
jupyter notebook notebooks/fraud_detection_workflow.ipynb
```

## 📊 Expected Results with Real Data

### Credit Card Fraud Dataset
- **Dataset Size**: ~284K transactions
- **Fraud Rate**: ~0.17% (492 fraudulent out of 284,807 total)
- **Expected AUC**: 0.95+ (with proper feature engineering)
- **Training Time**: 2-5 minutes (depending on hardware)

### Performance Comparison

| Model | AUC Score | Precision | Recall | F1-Score |
|-------|-----------|-----------|--------|----------|
| Logistic Regression | 0.85-0.90 | 0.70-0.80 | 0.80-0.90 | 0.75-0.85 |
| Random Forest | 0.90-0.95 | 0.80-0.90 | 0.85-0.95 | 0.85-0.90 |
| XGBoost | 0.95-0.98 | 0.85-0.95 | 0.90-0.98 | 0.90-0.95 |
| LightGBM | 0.95-0.98 | 0.85-0.95 | 0.90-0.98 | 0.90-0.95 |

## 🔧 Customization for Different Datasets

### For IEEE-CIS Dataset
```python
# Update data paths
data_path = "data/raw/train_transaction.csv"
fraud_column = "isFraud"

# Handle categorical features
categorical_features = ['ProductCD', 'card4', 'card6', 'P_emaildomain']
```

### For PaySim Dataset
```python
# Update data paths
data_path = "data/raw/PS_20174392719_1491204439457_log.csv"
fraud_column = "isFraud"

# Handle different feature names
amount_column = "amount"
```

## 🚀 Advanced Usage

### 1. Feature Engineering
```python
from src.feature_engineering import FraudFeatureEngineer

engineer = FraudFeatureEngineer()
df_enhanced = engineer.engineer_all_features(df, fraud_column='Class')
```

### 2. Hyperparameter Optimization
```python
from train_model import FraudModelTrainer

trainer = FraudModelTrainer()
best_params = trainer.hyperparameter_optimization('XGBoost', n_trials=100)
```

### 3. Model Deployment
```python
# Save best model
trainer.save_model('XGBoost', 'models/best_fraud_model.pkl')

# Load for predictions
trainer.load_model('XGBoost', 'models/best_fraud_model.pkl')
predictions, probabilities = trainer.predict_new_data('XGBoost', new_transactions)
```

## 📈 Performance Monitoring

### Key Metrics to Track
- **AUC-ROC**: Overall model performance
- **Precision**: Accuracy of fraud predictions
- **Recall**: Ability to catch all fraud
- **F1-Score**: Balanced measure
- **False Positive Rate**: Cost of false alarms

### Visualization Files Generated
- `feature_correlations.png`: Feature importance
- `roc_curves.png`: Model comparison
- `confusion_matrices.png`: Prediction accuracy
- `feature_importance.png`: Random Forest insights

## 🛠️ Troubleshooting

### Common Issues

1. **Kaggle API Errors**
   ```bash
   # Check credentials
   ls -la ~/.kaggle/kaggle.json
   
   # Test API
   kaggle datasets list --limit 1
   ```

2. **Memory Issues**
   ```python
   # Use smaller sample for testing
   df_sample = df.sample(n=10000, random_state=42)
   ```

3. **Package Installation**
   ```bash
   # Install missing packages
   pip3 install -r requirements.txt
   
   # For XGBoost on Mac
   brew install libomp
   ```

### Performance Tips

1. **Use Sample Data for Development**
   ```python
   # Use 10% of data for quick testing
   df_dev = df.sample(frac=0.1, random_state=42)
   ```

2. **Optimize Memory Usage**
   ```python
   # Use appropriate dtypes
   df['Amount'] = df['Amount'].astype('float32')
   ```

3. **Parallel Processing**
   ```python
   # Use all CPU cores
   model = RandomForestClassifier(n_jobs=-1)
   ```

## 📚 Additional Resources

- **Kaggle Notebooks**: Search for "credit card fraud" on Kaggle
- **Research Papers**: IEEE-CIS fraud detection competition papers
- **Community**: Kaggle forums and discussions

## 🎉 Success Indicators

You'll know everything is working when you see:
- ✅ Dataset loaded successfully
- ✅ AUC scores > 0.90
- ✅ Generated visualization files
- ✅ Model saved to `models/` directory
- ✅ No errors in the pipeline

---

**Happy Fraud Detection! 🕵️‍♂️**

For questions or issues, check the main README.md or open an issue in the repository. 