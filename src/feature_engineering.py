import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FraudFeatureEngineer:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.feature_stats = {}
        
    def create_time_features(self, df, time_column='Time'):
        """Create time-based features from transaction timestamps."""
        if time_column not in df.columns:
            print(f"Time column '{time_column}' not found. Skipping time features.")
            return df
        
        print("Creating time-based features...")
        
        # Convert time to datetime-like features
        df['hour'] = df[time_column] % 24
        df['day_of_week'] = (df[time_column] // 24) % 7
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        return df
    
    def create_amount_features(self, df, amount_column='Amount'):
        """Create amount-based features."""
        if amount_column not in df.columns:
            print(f"Amount column '{amount_column}' not found. Skipping amount features.")
            return df
        
        print("Creating amount-based features...")
        
        # Basic amount features
        df['amount_log'] = np.log1p(df[amount_column])
        df['amount_sqrt'] = np.sqrt(df[amount_column])
        df['amount_squared'] = df[amount_column] ** 2
        
        # Amount bins
        df['amount_bin'] = pd.cut(df[amount_column], bins=10, labels=False)
        
        # High-value transaction flag
        amount_95th = df[amount_column].quantile(0.95)
        df['is_high_value'] = (df[amount_column] > amount_95th).astype(int)
        
        return df
    
    def create_statistical_features(self, df, feature_columns):
        """Create statistical features from existing features."""
        print("Creating statistical features...")
        
        # Calculate rolling statistics for numerical features
        numerical_features = df[feature_columns].select_dtypes(include=[np.number]).columns
        
        for feature in numerical_features[:10]:  # Limit to first 10 features to avoid explosion
            # Rolling mean and std (if we have enough data)
            if len(df) > 100:
                df[f'{feature}_rolling_mean'] = df[feature].rolling(window=100, min_periods=1).mean()
                df[f'{feature}_rolling_std'] = df[feature].rolling(window=100, min_periods=1).std()
            
            # Z-score
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            if std_val > 0:
                df[f'{feature}_zscore'] = (df[feature] - mean_val) / std_val
            
            # Percentile rank
            df[f'{feature}_percentile'] = df[feature].rank(pct=True)
        
        return df
    
    def create_interaction_features(self, df, feature_columns):
        """Create interaction features between important features."""
        print("Creating interaction features...")
        
        # Select top features for interactions (limit to avoid explosion)
        top_features = feature_columns[:5]
        
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                if feat1 in df.columns and feat2 in df.columns:
                    # Multiplication
                    df[f'{feat1}_{feat2}_mult'] = df[feat1] * df[feat2]
                    
                    # Division (with safety check)
                    if df[feat2].abs().min() > 1e-8:
                        df[f'{feat1}_{feat2}_div'] = df[feat1] / (df[feat2] + 1e-8)
        
        return df
    
    def create_anomaly_features(self, df, feature_columns):
        """Create anomaly detection features."""
        print("Creating anomaly features...")
        
        # Calculate Mahalanobis distance for multivariate outliers
        numerical_features = df[feature_columns].select_dtypes(include=[np.number]).columns[:10]
        
        if len(numerical_features) > 1:
            # Calculate covariance matrix
            cov_matrix = df[numerical_features].cov()
            
            # Calculate Mahalanobis distance
            try:
                inv_cov_matrix = np.linalg.inv(cov_matrix.values)
                mean_vector = df[numerical_features].mean().values
                
                mahal_distances = []
                for idx, row in df[numerical_features].iterrows():
                    diff = row.values - mean_vector
                    mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                    mahal_distances.append(mahal_dist)
                
                df['mahalanobis_distance'] = mahal_distances
                df['is_anomaly'] = (df['mahalanobis_distance'] > df['mahalanobis_distance'].quantile(0.99)).astype(int)
            except:
                print("Could not calculate Mahalanobis distance")
        
        # Isolation Forest features (simplified)
        for feature in feature_columns[:5]:
            if feature in df.columns:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df[f'{feature}_is_outlier'] = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).astype(int)
        
        return df
    
    def apply_pca(self, df, feature_columns, n_components=10):
        """Apply PCA for dimensionality reduction."""
        print(f"Applying PCA with {n_components} components...")
        
        numerical_features = df[feature_columns].select_dtypes(include=[np.number]).columns
        
        if len(numerical_features) > n_components:
            # Scale features first
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[numerical_features])
            
            # Apply PCA
            self.pca = PCA(n_components=n_components)
            pca_features = self.pca.fit_transform(scaled_features)
            
            # Add PCA features to dataframe
            for i in range(n_components):
                df[f'pca_component_{i+1}'] = pca_features[:, i]
            
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return df
    
    def create_aggregate_features(self, df, group_columns=None):
        """Create aggregate features based on grouping."""
        if group_columns is None:
            return df
        
        print("Creating aggregate features...")
        
        # Example: if we have merchant or customer IDs
        for group_col in group_columns:
            if group_col in df.columns:
                # Transaction count per group
                group_counts = df.groupby(group_col).size().reset_index(name=f'{group_col}_transaction_count')
                df = df.merge(group_counts, on=group_col, how='left')
                
                # Average amount per group
                group_amounts = df.groupby(group_col)['Amount'].agg(['mean', 'std']).reset_index()
                group_amounts.columns = [group_col, f'{group_col}_avg_amount', f'{group_col}_std_amount']
                df = df.merge(group_amounts, on=group_col, how='left')
        
        return df
    
    def engineer_all_features(self, df, fraud_column=None):
        """Apply all feature engineering steps."""
        print("=" * 50)
        print("FEATURE ENGINEERING")
        print("=" * 50)
        
        original_shape = df.shape
        print(f"Original shape: {original_shape}")
        
        # Identify feature columns (exclude target)
        feature_columns = [col for col in df.columns if col != fraud_column]
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create amount features
        df = self.create_amount_features(df)
        
        # Create statistical features
        df = self.create_statistical_features(df, feature_columns)
        
        # Create interaction features
        df = self.create_interaction_features(df, feature_columns)
        
        # Create anomaly features
        df = self.create_anomaly_features(df, feature_columns)
        
        # Apply PCA
        df = self.apply_pca(df, feature_columns)
        
        # Handle missing values created by feature engineering
        df = df.fillna(df.median())
        
        print(f"Final shape: {df.shape}")
        print(f"Added {df.shape[1] - original_shape[1]} new features")
        
        return df
    
    def get_feature_importance_ranking(self, df, fraud_column):
        """Rank features by their correlation with fraud."""
        if fraud_column not in df.columns:
            return []
        
        # Calculate correlations with fraud
        feature_columns = [col for col in df.columns if col != fraud_column]
        correlations = df[feature_columns].corrwith(df[fraud_column]).abs().sort_values(ascending=False)
        
        return correlations
    
    def select_top_features(self, df, fraud_column, n_features=50):
        """Select top features based on correlation with fraud."""
        correlations = self.get_feature_importance_ranking(df, fraud_column)
        top_features = correlations.head(n_features).index.tolist()
        
        # Always include the fraud column
        if fraud_column not in top_features:
            top_features.append(fraud_column)
        
        return df[top_features]

def main():
    """Example usage of feature engineering."""
    # This would be used after loading data
    print("Feature Engineering Module")
    print("Use this module to enhance your fraud detection features.")
    print("\nExample usage:")
    print("engineer = FraudFeatureEngineer()")
    print("df_enhanced = engineer.engineer_all_features(df, fraud_column='Class')")

if __name__ == "__main__":
    main() 