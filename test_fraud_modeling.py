#!/usr/bin/env python3
"""
Comprehensive Fraud Detection Model Testing
Tests the complete pipeline using the sample dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def test_data_loading():
    """Test data loading and basic exploration."""
    print("=" * 60)
    print("1. DATA LOADING AND EXPLORATION")
    print("=" * 60)
    
    # Try to load the real dataset first
    real_data_path = "data/raw/creditcard.csv"
    sample_data_path = "data/raw/creditcard_sample.csv"
    
    if Path(real_data_path).exists():
        print("✓ Found real Kaggle dataset!")
        df = pd.read_csv(real_data_path)
        dataset_type = "Real Kaggle Dataset"
    elif Path(sample_data_path).exists():
        print("✓ Using sample dataset for demonstration")
        df = pd.read_csv(sample_data_path)
        dataset_type = "Sample Dataset"
    else:
        print("❌ No dataset found. Please download the dataset first.")
        print("Run: python3 download_kaggle_dataset.py")
        return None, None
    
    print(f"\nDataset: {dataset_type}")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Basic info
    print(f"\nColumns: {list(df.columns)}")
    print(f"Data types: {df.dtypes.value_counts()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Fraud analysis
    fraud_counts = df['Class'].value_counts()
    fraud_percentage = df['Class'].value_counts(normalize=True) * 100
    
    print(f"\nFraud Distribution:")
    print(f"  Legitimate: {fraud_counts[0]:,} ({fraud_percentage[0]:.2f}%)")
    print(f"  Fraudulent: {fraud_counts[1]:,} ({fraud_percentage[1]:.2f}%)")
    
    return df, dataset_type

def test_feature_analysis(df):
    """Test feature analysis and visualization."""
    print("\n" + "=" * 60)
    print("2. FEATURE ANALYSIS")
    print("=" * 60)
    
    # Select numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('Class')
    
    print(f"Analyzing {len(numerical_features)} numerical features...")
    
    # Calculate correlations with fraud
    correlations = df[numerical_features].corrwith(df['Class']).abs().sort_values(ascending=False)
    
    print(f"\nTop 10 features by correlation with fraud:")
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"  {i:2d}. {feature}: {corr:.4f}")
    
    # Create correlation plot
    plt.figure(figsize=(12, 6))
    correlations.head(10).plot(kind='bar')
    plt.title('Top 10 Features by Correlation with Fraud')
    plt.xlabel('Features')
    plt.ylabel('Absolute Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved feature correlation plot: feature_correlations.png")
    
    return correlations

def test_model_training(df):
    """Test model training with simplified approach."""
    print("\n" + "=" * 60)
    print("3. MODEL TRAINING")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('Class')
    
    X = df[numerical_features]
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Training fraud rate: {y_train.mean():.4f}")
    print(f"Test fraud rate: {y_test.mean():.4f}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'auc': auc_score
        }
        
        print(f"  AUC Score: {auc_score:.4f}")
        
        # Classification report
        print(f"  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))
    
    return results, X_test_scaled, y_test

def test_model_evaluation(results, X_test, y_test):
    """Test model evaluation and visualization."""
    print("\n" + "=" * 60)
    print("4. MODEL EVALUATION")
    print("=" * 60)
    
    from sklearn.metrics import roc_curve, confusion_matrix
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc"]:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved ROC curves plot: roc_curves.png")
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraudulent'],
                   yticklabels=['Legitimate', 'Fraudulent'],
                   ax=axes[i])
        axes[i].set_title(f'{name}\nConfusion Matrix')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved confusion matrices plot: confusion_matrices.png")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['auc'])
    print(f"\nBest performing model: {best_model}")
    print(f"Best AUC Score: {results[best_model]['auc']:.4f}")
    
    return best_model

def test_feature_importance(results, df):
    """Test feature importance analysis."""
    print("\n" + "=" * 60)
    print("5. FEATURE IMPORTANCE")
    print("=" * 60)
    
    # Get feature importance from Random Forest
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance = rf_model.feature_importances_
        
        # Get feature names
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features.remove('Class')
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': numerical_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        importance_df.head(15).plot(x='feature', y='importance', kind='barh')
        plt.title('Top 15 Feature Importance (Random Forest)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved feature importance plot: feature_importance.png")

def main():
    """Main testing function."""
    print("Fraud Detection Model Testing")
    print("=" * 60)
    
    # Test data loading
    df, dataset_type = test_data_loading()
    if df is None:
        return
    
    # Test feature analysis
    correlations = test_feature_analysis(df)
    
    # Test model training
    results, X_test, y_test = test_model_training(df)
    
    # Test model evaluation
    best_model = test_model_evaluation(results, X_test, y_test)
    
    # Test feature importance
    test_feature_importance(results, df)
    
    # Summary
    print("\n" + "=" * 60)
    print("TESTING SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset_type}")
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['Class'].mean():.4f}")
    print(f"Best model: {best_model}")
    print(f"Best AUC: {results[best_model]['auc']:.4f}")
    
    print(f"\nGenerated files:")
    print(f"  - feature_correlations.png")
    print(f"  - roc_curves.png")
    print(f"  - confusion_matrices.png")
    print(f"  - feature_importance.png")
    
    print(f"\n🎉 Testing completed successfully!")
    print(f"\nNext steps:")
    print(f"1. Download real Kaggle dataset: python3 download_kaggle_dataset.py")
    print(f"2. Run full exploration: python3 data_exploration.py")
    print(f"3. Run full training: python3 train_model.py")
    print(f"4. Interactive analysis: jupyter notebook notebooks/fraud_detection_workflow.ipynb")

if __name__ == "__main__":
    main() 