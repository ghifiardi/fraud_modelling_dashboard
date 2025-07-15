import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FraudModelTrainer:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.fraud_column = None
        self.feature_columns = None
        
    def load_data(self, data_path=None):
        """Load and prepare the dataset."""
        if data_path:
            self.data_path = data_path
        elif self.data_path:
            data_path = self.data_path
        else:
            raise ValueError("Please provide a data path")
            
        print(f"Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        
        # Identify fraud column
        self.identify_fraud_column()
        
        return self.df
    
    def identify_fraud_column(self):
        """Identify the fraud column in the dataset."""
        possible_fraud_columns = ['fraud', 'is_fraud', 'fraudulent', 'target', 'class', 'label']
        
        for col in possible_fraud_columns:
            if col in self.df.columns:
                self.fraud_column = col
                print(f"Fraud column identified: {col}")
                break
                
        if not self.fraud_column:
            # Look for binary columns
            binary_columns = []
            for col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    unique_vals = self.df[col].unique()
                    if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
                        binary_columns.append(col)
            
            if binary_columns:
                self.fraud_column = binary_columns[0]
                print(f"Using potential fraud column: {self.fraud_column}")
            else:
                raise ValueError("No fraud column identified. Please specify manually.")
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess the data for training."""
        print("=" * 50)
        print("DATA PREPROCESSING")
        print("=" * 50)
        
        # Select numerical features
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.fraud_column in numerical_features:
            numerical_features.remove(self.fraud_column)
        
        self.feature_columns = numerical_features
        print(f"Using {len(self.feature_columns)} numerical features")
        
        # Prepare features and target
        X = self.df[self.feature_columns]
        y = self.df[self.fraud_column]
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            X = X.fillna(X.median())
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set fraud rate: {self.y_train.mean():.4f}")
        print(f"Test set fraud rate: {self.y_test.mean():.4f}")
        
        # Scale features
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(20, len(self.feature_columns)))
        self.X_train_selected = self.feature_selector.fit_transform(self.X_train_scaled, self.y_train)
        self.X_test_selected = self.feature_selector.transform(self.X_test_scaled)
        
        selected_features = self.feature_selector.get_support()
        selected_feature_names = [self.feature_columns[i] for i in range(len(self.feature_columns)) if selected_features[i]]
        print(f"Selected {len(selected_feature_names)} features: {selected_feature_names}")
        
        return self.X_train_selected, self.X_test_selected, self.y_train, self.y_test
    
    def handle_imbalance(self, method='smote'):
        """Handle class imbalance in the training data."""
        print(f"Handling class imbalance using {method.upper()}...")
        
        if method.lower() == 'smote':
            sampler = SMOTE(random_state=42)
        elif method.lower() == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif method.lower() == 'smoteenn':
            sampler = SMOTEENN(random_state=42)
        elif method.lower() == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            print("No resampling applied")
            return self.X_train_selected, self.y_train
        
        X_resampled, y_resampled = sampler.fit_resample(self.X_train_selected, self.y_train)
        
        print(f"Original training set: {len(self.y_train)} samples")
        print(f"Resampled training set: {len(y_resampled)} samples")
        print(f"Original fraud rate: {self.y_train.mean():.4f}")
        print(f"Resampled fraud rate: {y_resampled.mean():.4f}")
        
        return X_resampled, y_resampled
    
    def train_models(self, handle_imbalance=True):
        """Train multiple fraud detection models."""
        print("=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        if handle_imbalance:
            X_train, y_train = self.handle_imbalance('smote')
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc': auc_score
            }
            
            print(f"{name} - AUC: {auc_score:.4f}")
            
            # Store model
            self.models[name] = model
        
        return results
    
    def hyperparameter_optimization(self, model_name='XGBoost', n_trials=50):
        """Optimize hyperparameters using Optuna."""
        print(f"Optimizing hyperparameters for {model_name}...")
        
        def objective(trial):
            if model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                model = xgb.XGBClassifier(**params)
            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
            else:
                return 0.0
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_selected, self.y_train, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                      scoring='roc_auc')
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best {model_name} parameters: {study.best_params}")
        print(f"Best CV AUC: {study.best_value:.4f}")
        
        return study.best_params
    
    def evaluate_models(self, results):
        """Evaluate and compare model performance."""
        print("=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print detailed results
        for name, result in results.items():
            print(f"\n{name} Results:")
            print("-" * 30)
            print(f"AUC Score: {result['auc']:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, result['predictions']))
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, result['predictions'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Legitimate', 'Fraudulent'],
                       yticklabels=['Legitimate', 'Fraudulent'])
            plt.title(f'{name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
    
    def save_model(self, model_name, filepath):
        """Save the trained model."""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"Model saved to {filepath}")
        else:
            print(f"Model {model_name} not found")
    
    def load_model(self, model_name, filepath):
        """Load a saved model."""
        self.models[model_name] = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_new_data(self, model_name, new_data):
        """Make predictions on new data."""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        # Preprocess new data
        new_data_scaled = self.scaler.transform(new_data[self.feature_columns])
        new_data_selected = self.feature_selector.transform(new_data_scaled)
        
        # Make predictions
        predictions = self.models[model_name].predict(new_data_selected)
        probabilities = self.models[model_name].predict_proba(new_data_selected)[:, 1]
        
        return predictions, probabilities

def main():
    """Main function to run model training."""
    # Example usage
    sample_data_path = "data/raw/creditcard.csv"
    
    trainer = FraudModelTrainer()
    
    try:
        # Load data
        trainer.load_data(sample_data_path)
        
        # Train models
        results = trainer.train_models(handle_imbalance=True)
        
        # Evaluate models
        trainer.evaluate_models(results)
        
        # Save best model
        best_model = max(results.keys(), key=lambda x: results[x]['auc'])
        trainer.save_model(best_model, f"models/{best_model.lower().replace(' ', '_')}_model.pkl")
        
        print(f"\nBest model: {best_model} with AUC: {results[best_model]['auc']:.4f}")
        
    except FileNotFoundError:
        print(f"Dataset not found at {sample_data_path}")
        print("Please download a fraud dataset and update the path.")
        print("Recommended: Credit Card Fraud Detection from Kaggle")

if __name__ == "__main__":
    main()
