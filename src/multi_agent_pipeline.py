"""
Multi-Agent Fraud Detection Pipeline with Context Passing (LangGraph)
- Demonstrates agent-to-agent communication with context passing
- Each agent adds to or updates the context dict
- Easily extensible for real-world integration
"""

from langgraph.graph import StateGraph, END
import pandas as pd
import random

# --- Agent Definitions ---

def risk_scoring_agent(context):
    # Dummy risk scoring logic (replace with real model)
    context['risk_score'] = random.randint(0, 10)
    context['risk_level'] = 'HIGH_RISK' if context['risk_score'] > 7 else 'MEDIUM_RISK' if context['risk_score'] > 3 else 'LOW_RISK'
    context['risk_probability'] = random.random()
    context['recommended_action'] = 'BLOCK_TRANSACTION' if context['risk_level'] == 'HIGH_RISK' else 'REQUIRE_ADDITIONAL_VERIFICATION' if context['risk_level'] == 'MEDIUM_RISK' else 'ALLOW_TRANSACTION'
    return context

def enrichment_agent_factory(kaggle_df):
    def enrichment_agent(context):
        card_id = context.get('customer_id')
        if card_id is not None and 'customer_id' in kaggle_df.columns:
            customer_history = kaggle_df[kaggle_df['customer_id'] == card_id]
            if not customer_history.empty:
                context['kaggle_avg_amount'] = customer_history['amount'].mean()
                context['kaggle_tx_count'] = len(customer_history)
                context['kaggle_fraud_rate'] = customer_history['is_fraud'].mean()
            else:
                context['kaggle_avg_amount'] = None
                context['kaggle_tx_count'] = 0
                context['kaggle_fraud_rate'] = None
        else:
            context['kaggle_avg_amount'] = None
            context['kaggle_tx_count'] = 0
            context['kaggle_fraud_rate'] = None
        context['enriched'] = True
        context['enrichment_details'] = 'Kaggle-based enrichment complete.'
        return context
    return enrichment_agent

def external_api_client_agent(context):
    # Simulate an external API call (e.g., device reputation)
    context['device_reputation'] = random.choice(['trusted', 'unknown', 'suspicious'])
    context['external_api_details'] = 'Device reputation checked.'
    return context

def explanation_agent(context):
    context['explanation'] = (
        f"Transaction {context.get('transaction_id', 'N/A')} risk: {context.get('risk_level', 'N/A')}, device: {context.get('device_reputation', 'N/A')}"
    )
    return context

def escalation_agent(context):
    if context.get('risk_level') == 'HIGH_RISK' or context.get('device_reputation') == 'suspicious':
        context['escalated'] = True
        context['escalation_reason'] = 'High risk or suspicious device.'
    else:
        context['escalated'] = False
    return context

# --- Pipeline Orchestration ---
def main():
    # Load Kaggle dataset and prepare for enrichment
    kaggle_df = pd.read_csv("data/creditcard_2023.csv")
    kaggle_df = kaggle_df.rename(columns={"Amount": "amount", "Class": "is_fraud"})
    kaggle_df['customer_id'] = kaggle_df.index // 10

    # Build the LangGraph pipeline
    state_schema = dict
    graph = StateGraph(state_schema)
    graph.add_node('risk', risk_scoring_agent)
    graph.add_node('enrich', enrichment_agent_factory(kaggle_df))
    graph.add_node('external', external_api_client_agent)
    graph.add_node('explain', explanation_agent)
    graph.add_node('escalate', escalation_agent)
    graph.add_edge('risk', 'enrich')
    graph.add_edge('enrich', 'external')
    graph.add_edge('external', 'explain')
    graph.add_edge('explain', 'escalate')
    graph.add_edge('escalate', END)
    graph.set_entry_point('risk')
    app = graph.compile()

    # Example transaction context
    sample_tx = {
        'transaction_id': 55555,
        'customer_id': 1,
        'amount': 1200,
        'device_type': 'MOBILE',
        'location': 'INTERNATIONAL',
        # ... add more fields as needed ...
    }
    result = app.invoke(sample_tx)
    print("\nFinal context after multi-agent pipeline:")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main() 