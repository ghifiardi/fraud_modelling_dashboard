import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
import numpy as np

class FraudDataDownloader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_creditcard_fraud(self):
        """Download Credit Card Fraud Detection dataset from Kaggle."""
        print("Downloading Credit Card Fraud Detection dataset...")
        
        # Kaggle API credentials needed
        # You need to set up kaggle.json in ~/.kaggle/kaggle.json
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'mlg-ulb/creditcardfraud',
                path=self.data_dir,
                unzip=True
            )
            print("Credit Card Fraud dataset downloaded successfully!")
            return True
        except Exception as e:
            print(f"Error downloading from Kaggle: {e}")
            print("Please install kaggle package and set up credentials:")
            print("pip install kaggle")
            print("Then download kaggle.json from your Kaggle account settings")
            return False
    
    def create_sample_dataset(self):
        """Create a synthetic sample dataset for testing."""
        print("Creating synthetic fraud dataset for testing...")
        
        np.random.seed(42)
        n_samples = 10000
        n_features = 30
        
        # Generate synthetic features
        features = {}
        for i in range(n_features):
            if i < 28:  # V1-V28 (anonymized features)
                features[f'V{i+1}'] = np.random.normal(0, 1, n_samples)
            elif i == 28:  # Amount
                features['Amount'] = np.random.exponential(100, n_samples)
            else:  # Time
                features['Time'] = np.random.uniform(0, 172792, n_samples)
        
        # Create fraud labels (imbalanced dataset)
        fraud_rate = 0.0017  # ~0.17% fraud rate
        fraud_labels = np.random.binomial(1, fraud_rate, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(features)
        df['Class'] = fraud_labels
        
        # Save to file
        output_path = self.data_dir / "creditcard_sample.csv"
        df.to_csv(output_path, index=False)
        print(f"Sample dataset created: {output_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud rate: {fraud_labels.mean():.4f}")
        
        return output_path
    
    def download_from_url(self, url, filename):
        """Download dataset from URL."""
        print(f"Downloading {filename} from {url}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            file_path = self.data_dir / filename
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None
    
    def list_available_datasets(self):
        """List available fraud datasets."""
        datasets = {
            "creditcard": {
                "name": "Credit Card Fraud Detection",
                "description": "European credit card transactions with fraud labels",
                "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
                "features": "30 features (28 anonymized + Amount + Time)",
                "fraud_rate": "0.17%"
            },
            "ieee": {
                "name": "IEEE-CIS Fraud Detection",
                "description": "Large-scale fraud detection dataset",
                "url": "https://www.kaggle.com/c/ieee-fraud-detection/data",
                "features": "400+ features",
                "fraud_rate": "3.5%"
            },
            "paysim": {
                "name": "Synthetic Financial Dataset (PaySim)",
                "description": "Synthetic mobile money transactions",
                "url": "https://www.kaggle.com/datasets/ealaxi/paysim1",
                "features": "11 features",
                "fraud_rate": "0.6%"
            }
        }
        
        print("Available Fraud Detection Datasets:")
        print("=" * 50)
        for key, info in datasets.items():
            print(f"\n{key.upper()}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Features: {info['features']}")
            print(f"  Fraud Rate: {info['fraud_rate']}")
            print(f"  URL: {info['url']}")
        
        return datasets

def main():
    """Main function to download datasets."""
    downloader = FraudDataDownloader()
    
    print("Fraud Detection Dataset Downloader")
    print("=" * 40)
    
    # List available datasets
    datasets = downloader.list_available_datasets()
    
    print("\n" + "=" * 40)
    print("DOWNLOAD OPTIONS:")
    print("1. Download from Kaggle (requires kaggle API)")
    print("2. Create synthetic sample dataset")
    print("3. Manual download instructions")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        success = downloader.download_creditcard_fraud()
        if not success:
            print("\nManual download instructions:")
            print("1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            print("2. Download the dataset")
            print("3. Extract to data/raw/ directory")
            print("4. Rename to creditcard.csv")
    
    elif choice == "2":
        downloader.create_sample_dataset()
        print("\nSample dataset created! You can now run the exploration and training scripts.")
    
    elif choice == "3":
        print("\nManual download instructions:")
        print("1. Visit the Kaggle dataset URLs above")
        print("2. Download the CSV files")
        print("3. Place them in the data/raw/ directory")
        print("4. Update the file paths in data_exploration.py and train_model.py")
    
    else:
        print("Invalid choice. Creating sample dataset...")
        downloader.create_sample_dataset()

if __name__ == "__main__":
    main() 