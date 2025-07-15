import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FraudDataExplorer:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.fraud_column = None
        
    def load_data(self, data_path=None):
        """Load transaction data from CSV file."""
        if data_path:
            self.data_path = data_path
        elif self.data_path:
            data_path = self.data_path
        else:
            raise ValueError("Please provide a data path")
            
        print(f"Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        return self.df
    
    def identify_fraud_column(self):
        """Identify the fraud column in the dataset."""
        possible_fraud_columns = ['fraud', 'is_fraud', 'fraudulent', 'target', 'class', 'label']
        
        for col in possible_fraud_columns:
            if col in self.df.columns:
                self.fraud_column = col
                print(f"Fraud column identified: {col}")
                return col
                
        # If no standard name found, look for binary columns
        binary_columns = []
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                unique_vals = self.df[col].unique()
                if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
                    binary_columns.append(col)
        
        if binary_columns:
            print(f"Potential fraud columns found: {binary_columns}")
            self.fraud_column = binary_columns[0]
            return binary_columns[0]
        
        print("No fraud column identified. Please specify manually.")
        return None
    
    def basic_info(self):
        """Display basic information about the dataset."""
        print("=" * 50)
        print("BASIC DATASET INFORMATION")
        print("=" * 50)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Number of features: {self.df.shape[1]}")
        print(f"Number of samples: {self.df.shape[0]}")
        
        print("\nColumn names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\nData types:")
        print(self.df.dtypes.value_counts())
        
        print("\nMissing values:")
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print(missing_data[missing_data > 0])
        else:
            print("No missing values found!")
    
    def fraud_analysis(self):
        """Analyze fraud distribution and patterns."""
        if not self.fraud_column:
            self.identify_fraud_column()
        
        if not self.fraud_column:
            print("Cannot perform fraud analysis without identifying fraud column")
            return
        
        print("=" * 50)
        print("FRAUD ANALYSIS")
        print("=" * 50)
        
        fraud_counts = self.df[self.fraud_column].value_counts()
        fraud_percentage = self.df[self.fraud_column].value_counts(normalize=True) * 100
        
        print(f"Fraud distribution:")
        print(f"Legitimate transactions: {fraud_counts[0]:,} ({fraud_percentage[0]:.2f}%)")
        print(f"Fraudulent transactions: {fraud_counts[1]:,} ({fraud_percentage[1]:.2f}%)")
        print(f"Total transactions: {len(self.df):,}")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        fraud_counts.plot(kind='bar', ax=ax1, color=['green', 'red'])
        ax1.set_title('Transaction Distribution')
        ax1.set_xlabel('Transaction Type')
        ax1.set_ylabel('Count')
        ax1.set_xticklabels(['Legitimate', 'Fraudulent'])
        
        # Pie chart
        ax2.pie(fraud_counts.values, labels=['Legitimate', 'Fraudulent'], 
                autopct='%1.2f%%', colors=['green', 'red'])
        ax2.set_title('Transaction Distribution (%)')
        
        plt.tight_layout()
        plt.show()
    
    def feature_analysis(self, n_features=10):
        """Analyze numerical features and their relationship with fraud."""
        if not self.fraud_column:
            self.identify_fraud_column()
        
        if not self.fraud_column:
            print("Cannot perform feature analysis without identifying fraud column")
            return
        
        print("=" * 50)
        print("FEATURE ANALYSIS")
        print("=" * 50)
        
        # Select numerical features
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.fraud_column in numerical_features:
            numerical_features.remove(self.fraud_column)
        
        print(f"Analyzing {len(numerical_features)} numerical features...")
        
        # Calculate statistics by fraud status
        fraud_stats = self.df.groupby(self.fraud_column)[numerical_features].agg(['mean', 'std', 'min', 'max'])
        
        # Create subplots for feature distributions
        n_cols = 3
        n_rows = (len(numerical_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(numerical_features[:n_features]):
            row = i // n_cols
            col = i % n_cols
            
            # Plot distribution by fraud status
            self.df[self.df[self.fraud_column] == 0][feature].hist(
                alpha=0.7, label='Legitimate', ax=axes[row, col], bins=30, color='green')
            self.df[self.df[self.fraud_column] == 1][feature].hist(
                alpha=0.7, label='Fraudulent', ax=axes[row, col], bins=30, color='red')
            
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].legend()
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numerical_features[:n_features]), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return fraud_stats
    
    def correlation_analysis(self):
        """Analyze correlations between features and fraud."""
        if not self.fraud_column:
            self.identify_fraud_column()
        
        if not self.fraud_column:
            print("Cannot perform correlation analysis without identifying fraud column")
            return
        
        print("=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Calculate correlations with fraud
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.fraud_column in numerical_features:
            numerical_features.remove(self.fraud_column)
        
        correlations = self.df[numerical_features].corrwith(self.df[self.fraud_column]).abs().sort_values(ascending=False)
        
        print("Top 10 features most correlated with fraud:")
        print(correlations.head(10))
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.df[numerical_features + [self.fraud_column]].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return correlations
    
    def time_series_analysis(self, time_column=None):
        """Analyze fraud patterns over time if time column exists."""
        if time_column and time_column in self.df.columns:
            print("=" * 50)
            print("TIME SERIES ANALYSIS")
            print("=" * 50)
            
            # Convert to datetime if needed
            if self.df[time_column].dtype == 'object':
                self.df[time_column] = pd.to_datetime(self.df[time_column])
            
            # Group by time and calculate fraud rate
            self.df['date'] = self.df[time_column].dt.date
            daily_fraud_rate = self.df.groupby('date')[self.fraud_column].mean()
            
            plt.figure(figsize=(15, 6))
            daily_fraud_rate.plot(kind='line', marker='o')
            plt.title('Daily Fraud Rate Over Time')
            plt.xlabel('Date')
            plt.ylabel('Fraud Rate')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def generate_report(self, save_path=None):
        """Generate a comprehensive data exploration report."""
        print("=" * 50)
        print("GENERATING COMPREHENSIVE DATA EXPLORATION REPORT")
        print("=" * 50)
        
        self.basic_info()
        self.fraud_analysis()
        fraud_stats = self.feature_analysis()
        correlations = self.correlation_analysis()
        
        # Try to identify time column
        time_columns = [col for col in self.df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_columns:
            self.time_series_analysis(time_columns[0])
        
        print("\n" + "=" * 50)
        print("EXPLORATION COMPLETE!")
        print("=" * 50)
        
        return {
            'fraud_stats': fraud_stats,
            'correlations': correlations,
            'shape': self.df.shape,
            'fraud_column': self.fraud_column
        }

def main():
    """Main function to run data exploration."""
    # Example usage with a sample dataset
    # You can replace this with your actual dataset path
    sample_data_path = "data/raw/creditcard_sample.csv"
    
    explorer = FraudDataExplorer()
    
    try:
        # Try to load the sample dataset
        explorer.load_data(sample_data_path)
        explorer.generate_report()
    except FileNotFoundError:
        print(f"Dataset not found at {sample_data_path}")
        print("Please download a fraud dataset (e.g., from Kaggle) and update the path.")
        print("Recommended datasets:")
        print("1. Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. IEEE-CIS Fraud Detection: https://www.kaggle.com/c/ieee-fraud-detection/data")
        print("3. Synthetic Financial Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1")

if __name__ == "__main__":
    main()
