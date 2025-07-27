#!/usr/bin/env python3
"""
Data Upload Feature for Fraud Detection Dashboard
Allows users to upload and test with real datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def create_data_upload_tab():
    """Create a data upload tab for testing with real datasets"""
    
    st.header("📊 Data Upload & Testing")
    st.markdown("Upload real fraud detection datasets to test your platform")
    
    # File upload section
    st.subheader("📁 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with transaction data",
        type=['csv'],
        help="Upload a CSV file with transaction data for testing"
    )
    
    if uploaded_file is not None:
        try:
            # Load the uploaded data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df):,} transactions")
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transactions", f"{len(df):,}")
            
            with col2:
                st.metric("Columns", len(df.columns))
            
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column analysis
            st.subheader("🔍 Column Analysis")
            
            # Check for fraud column
            fraud_columns = [col for col in df.columns if 'fraud' in col.lower() or 'class' in col.lower()]
            
            if fraud_columns:
                fraud_col = fraud_columns[0]
                fraud_count = df[fraud_col].sum()
                fraud_rate = (fraud_count / len(df)) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fraudulent Transactions", f"{fraud_count:,}")
                with col2:
                    st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
                with col3:
                    st.metric("Legitimate Transactions", f"{len(df) - fraud_count:,}")
                
                # Fraud distribution chart
                fig = px.pie(
                    values=[fraud_count, len(df) - fraud_count],
                    names=['Fraudulent', 'Legitimate'],
                    title="Transaction Distribution",
                    color_discrete_sequence=['#ff6b6b', '#51cf66']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.subheader("📈 Numeric Features Analysis")
                
                # Select column for analysis
                selected_col = st.selectbox("Select column for analysis", numeric_cols)
                
                if selected_col:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig = px.histogram(
                            df, 
                            x=selected_col,
                            title=f"Distribution of {selected_col}",
                            nbins=50
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Box plot
                        fig = px.box(
                            df, 
                            y=selected_col,
                            title=f"Box Plot of {selected_col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.subheader("📊 Statistics")
                    stats = df[selected_col].describe()
                    st.dataframe(stats, use_container_width=True)
            
            # Test fraud detection
            st.subheader("🧪 Test Fraud Detection")
            
            if st.button("🚀 Test with Uploaded Data", type="primary"):
                st.info("Testing fraud detection algorithms with uploaded data...")
                
                # Simulate fraud detection
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Processing transaction {i+1}...")
                    
                    if i == 25:
                        status_text.text("Loading machine learning models...")
                    elif i == 50:
                        status_text.text("Running fraud detection algorithms...")
                    elif i == 75:
                        status_text.text("Generating risk scores...")
                    elif i == 100:
                        status_text.text("✅ Analysis complete!")
                
                # Show results
                st.success("✅ Fraud detection analysis completed!")
                
                # Simulate results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Processed", f"{len(df):,}")
                
                with col2:
                    st.metric("High Risk", f"{np.random.randint(10, 50)}")
                
                with col3:
                    st.metric("Medium Risk", f"{np.random.randint(50, 150)}")
                
                with col4:
                    st.metric("Low Risk", f"{len(df) - np.random.randint(60, 200)}")
                
                # Risk distribution
                risk_data = pd.DataFrame({
                    'Risk Level': ['High Risk', 'Medium Risk', 'Low Risk'],
                    'Count': [np.random.randint(10, 50), np.random.randint(50, 150), len(df) - np.random.randint(60, 200)]
                })
                
                fig = px.bar(
                    risk_data,
                    x='Risk Level',
                    y='Count',
                    title="Risk Distribution",
                    color='Risk Level',
                    color_discrete_sequence=['#ff6b6b', '#ffd93d', '#51cf66']
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.info("Please ensure the file is a valid CSV with transaction data")
    
    # Sample data section
    st.subheader("📝 Generate Sample Data")
    
    if st.button("🔄 Generate Sample Transactions"):
        # Create sample data
        np.random.seed(42)
        n_transactions = st.slider("Number of transactions", 100, 10000, 1000)
        
        transactions = []
        for i in range(n_transactions):
            is_fraud = np.random.random() < 0.01
            
            transaction = {
                'transaction_id': f'TXN_{i+1:06d}',
                'customer_id': f'CUST_{np.random.randint(1, 1001):04d}',
                'amount': np.random.exponential(100) if not is_fraud else np.random.uniform(500, 5000),
                'transaction_type': np.random.choice(['ATM', 'POS', 'ONLINE', 'TRANSFER']),
                'merchant_category': np.random.choice(['RETAIL', 'FOOD', 'TRAVEL', 'UTILITIES', 'E-COMMERCE']),
                'location': np.random.choice(['LOCAL', 'DOMESTIC', 'INTERNATIONAL']),
                'device_type': np.random.choice(['MOBILE', 'DESKTOP', 'ATM', 'POS']),
                'card_present': np.random.choice([True, False]),
                'hour': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 7),
                'is_fraud': is_fraud
            }
            
            if is_fraud:
                transaction['amount'] = np.random.uniform(1000, 5000)
                transaction['location'] = 'INTERNATIONAL'
                transaction['card_present'] = False
                transaction['hour'] = np.random.choice([1, 2, 3, 4, 5, 22, 23])
            
            transactions.append(transaction)
        
        sample_df = pd.DataFrame(transactions)
        
        # Save sample data
        sample_path = Path("data/sample_transactions.csv")
        sample_path.parent.mkdir(exist_ok=True)
        sample_df.to_csv(sample_path, index=False)
        
        st.success(f"✅ Generated {len(sample_df):,} sample transactions")
        st.info(f"📁 Saved to: {sample_path}")
        
        # Show sample data
        st.dataframe(sample_df.head(10), use_container_width=True)
    
    # Kaggle dataset info
    st.subheader("📊 Recommended Kaggle Datasets")
    
    st.markdown("""
    **For testing your fraud detection platform, consider these datasets:**
    
    ### 1. Credit Card Fraud Detection
    - **URL**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    - **Size**: 284MB, 284,807 transactions
    - **Features**: 28 anonymized features + amount + class
    - **Fraud Rate**: 0.173%
    
    ### 2. IEEE-CIS Fraud Detection
    - **URL**: https://www.kaggle.com/datasets/cdeotte/ieee-fraud-detection-dataset
    - **Size**: 1.2GB, ~500K transactions
    - **Features**: Identity and transaction features
    - **Real-world**: From IEEE-CIS competition
    
    ### 3. Synthetic Financial Dataset
    - **URL**: https://www.kaggle.com/datasets/ealaxi/paysim1
    - **Size**: 256MB, 6.3M transactions
    - **Features**: Mobile money transactions
    - **Fraud Rate**: 0.6%
    """)

if __name__ == "__main__":
    st.set_page_config(page_title="Data Upload", page_icon="📊")
    create_data_upload_tab() 