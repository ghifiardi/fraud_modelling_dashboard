"""
Fraud Detection with LangGraph AI Agent Enrichment
- Scores transactions using BankFraudDetector
- Enriches suspicious transactions with FraudEnrichmentAgent
- Prints results for both enriched and safe transactions
"""

import pandas as pd
from src.bank_fraud_detector import BankFraudDetector
from src.langgraph_agent import FraudEnrichmentAgent
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_enrichment_pipeline")

def process_and_enrich_transaction(tx: dict, detector, enrichment_agent):
    try:
        # Score the transaction
        result = detector.predict_transaction_risk(pd.DataFrame([tx]))
        risk_level = result['risk_level']
        tx_dict = tx.copy()
        tx_dict.update(result)
        if risk_level in ['HIGH_RISK', 'MEDIUM_RISK']:
            enriched = enrichment_agent.run(tx_dict)
            logger.info(f"Enriched suspicious transaction: {enriched}")
            return enriched
        else:
            logger.info(f"Safe transaction: {tx_dict}")
            return tx_dict
    except Exception as e:
        logger.error(f"Error processing transaction {tx.get('transaction_id', 'N/A')}: {e}")
        return None

def main():
    # 1. Load the trained fraud detection model
    detector = BankFraudDetector()
    detector.load_bank_model("models/bank_fraud_detector.pkl")

    # 2. Load Kaggle dataset for enrichment
    kaggle_df = pd.read_csv("data/creditcard_2023.csv")
    kaggle_df = kaggle_df.rename(columns={"Amount": "amount", "Class": "is_fraud"})
    kaggle_df['customer_id'] = kaggle_df.index // 10

    # 3. Initialize the enrichment agent with Kaggle data
    enrichment_agent = FraudEnrichmentAgent(kaggle_df)

    # 4. Example: Score a batch of new transactions (replace with your real data)
    new_transactions = pd.DataFrame([
        {
            'transaction_id': 10001,
            'customer_id': 1,  # Use a valid synthetic customer_id for enrichment
            'amount': 2500,
            'transaction_type': 'ONLINE',
            'merchant_category': 'RETAIL',
            'hour': 23,
            'day_of_week': 5,
            'location': 'INTERNATIONAL',
            'device_type': 'MOBILE',
            'card_present': False,
            'previous_fraud_flag': False,
            'account_age_days': 365,
            'balance_before': 5000,
            'balance_after': 2500,
            'is_fraud': False,  # Added for feature engineering
        },
        {
            'transaction_id': 10002,
            'customer_id': 2,  # Use a valid synthetic customer_id for enrichment
            'amount': 50,
            'transaction_type': 'ATM',
            'merchant_category': 'FOOD',
            'hour': 14,
            'day_of_week': 2,
            'location': 'LOCAL',
            'device_type': 'ATM',
            'card_present': True,
            'previous_fraud_flag': False,
            'account_age_days': 1200,
            'balance_before': 2000,
            'balance_after': 1950,
            'is_fraud': False,  # Added for feature engineering
        },
        # Add more transactions as needed
    ])

    # 4. Feature engineering (must match training pipeline)
    new_transactions = detector.engineer_bank_features(new_transactions)

    # 5. Process and enrich each transaction
    enriched_results = []
    for idx, tx in new_transactions.iterrows():
        enriched = process_and_enrich_transaction(tx.to_dict(), detector, enrichment_agent)
        enriched_results.append(enriched)

    # 6. Print all results
    print("\nAll processed transactions:")
    for res in enriched_results:
        print(res)

if __name__ == "__main__":
    main() 