"""
LangGraph AI Agent for Bank Fraud Detection Enrichment
- Enriches suspicious transactions with additional context from Kaggle dataset
- Provides explanations for alerts
- Supports human-in-the-loop review
"""

from langgraph.graph import StateGraph
from langgraph.graph import END
from typing import Dict, Any, Callable
import pandas as pd

# Enrichment function using Kaggle dataset
def enrich_transaction_with_kaggle(kaggle_df: pd.DataFrame) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def enrich(transaction: Dict[str, Any]) -> Dict[str, Any]:
        card_id = transaction.get('customer_id')
        if card_id is not None and 'customer_id' in kaggle_df.columns:
            customer_history = kaggle_df[kaggle_df['customer_id'] == card_id]
            if not customer_history.empty:
                transaction['kaggle_avg_amount'] = customer_history['amount'].mean()
                transaction['kaggle_tx_count'] = len(customer_history)
                transaction['kaggle_fraud_rate'] = customer_history['is_fraud'].mean()
            else:
                transaction['kaggle_avg_amount'] = None
                transaction['kaggle_tx_count'] = 0
                transaction['kaggle_fraud_rate'] = None
        else:
            transaction['kaggle_avg_amount'] = None
            transaction['kaggle_tx_count'] = 0
            transaction['kaggle_fraud_rate'] = None
        transaction['enriched'] = True
        transaction['enrichment_details'] = 'Kaggle-based enrichment complete.'
        return transaction
    return enrich

# Explanation function
def explain_alert(transaction: Dict[str, Any]) -> Dict[str, Any]:
    transaction['explanation'] = (
        f"Transaction {transaction.get('transaction_id', 'N/A')} flagged due to high risk score: {transaction.get('risk_score', 'N/A')}"
    )
    return transaction

# Human-in-the-loop function
def escalate_to_human(transaction: Dict[str, Any]) -> Dict[str, Any]:
    transaction['escalated'] = True
    transaction['escalation_reason'] = 'Requires analyst review.'
    return transaction

# Define the LangGraph workflow
class FraudEnrichmentAgent:
    def __init__(self, kaggle_df: pd.DataFrame):
        state_schema = dict
        self.graph = StateGraph(state_schema)
        self.graph.add_node('enrich', enrich_transaction_with_kaggle(kaggle_df))
        self.graph.add_node('explain', explain_alert)
        self.graph.add_node('escalate', escalate_to_human)
        self.graph.add_edge('enrich', 'explain')
        self.graph.add_edge('explain', 'escalate')
        self.graph.add_edge('escalate', END)
        self.graph.set_entry_point('enrich')
        self.app = self.graph.compile()

    def run(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        result = self.app.invoke(transaction)
        return result

# Example usage
def main():
    # Load Kaggle dataset (replace with your actual path and column names)
    kaggle_df = pd.read_csv("data/creditcard_2023.csv")
    # Rename columns to match enrichment logic
    kaggle_df = kaggle_df.rename(columns={"Amount": "amount", "Class": "is_fraud"})
    # Create a synthetic customer_id (every 10 transactions = 1 customer)
    kaggle_df['customer_id'] = kaggle_df.index // 10
    agent = FraudEnrichmentAgent(kaggle_df)
    sample_tx = {
        'transaction_id': 12345,
        'customer_id': 1,
        'amount': 2500,
        'risk_score': 8,
        'is_fraud': True
    }
    enriched = agent.run(sample_tx)
    print('Enriched Transaction:', enriched)

if __name__ == "__main__":
    main() 