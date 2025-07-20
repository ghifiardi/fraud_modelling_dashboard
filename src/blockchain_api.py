#!/usr/bin/env python3
"""
Blockchain API Server for Fraud Detection
Provides REST endpoints for blockchain operations and smart contract validation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import logging

# Import blockchain components
from blockchain_core import BlockchainManager, Transaction, TransactionStatus, RiskLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class BlockchainAPIServer:
    """Blockchain API server for fraud detection"""
    
    def __init__(self):
        self.blockchain_manager = BlockchainManager()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'blockchain_valid': self.blockchain_manager.blockchain.is_chain_valid()
            })
        
        @app.route('/api/blockchain/status', methods=['GET'])
        def get_blockchain_status():
            """Get blockchain status"""
            try:
                status = self.blockchain_manager.get_blockchain_status()
                
                # Convert enum values to strings for JSON serialization
                serializable_status = {
                    'chain': [],
                    'pending_transactions': [],
                    'difficulty': status['difficulty'],
                    'is_valid': status['is_valid'],
                    'statistics': status['statistics']
                }
                
                # Convert blocks
                for block in status['chain']:
                    serializable_block = {
                        'index': block['index'],
                        'timestamp': block['timestamp'],
                        'transactions': [],
                        'previous_hash': block['previous_hash'],
                        'nonce': block['nonce'],
                        'merkle_root': block['merkle_root'],
                        'block_hash': block['block_hash']
                    }
                    
                    # Convert transactions in block
                    for tx in block['transactions']:
                        serializable_tx = {
                            'transaction_id': tx['transaction_id'],
                            'customer_id': tx['customer_id'],
                            'amount': tx['amount'],
                            'merchant_id': tx['merchant_id'],
                            'timestamp': tx['timestamp'],
                            'location': tx['location'],
                            'payment_method': tx['payment_method'],
                            'risk_score': tx['risk_score'],
                            'risk_level': tx['risk_level'].value if hasattr(tx['risk_level'], 'value') else str(tx['risk_level']),
                            'fraud_probability': tx['fraud_probability'],
                            'status': tx['status'].value if hasattr(tx['status'], 'value') else str(tx['status']),
                            'metadata': tx['metadata']
                        }
                        serializable_block['transactions'].append(serializable_tx)
                    
                    serializable_status['chain'].append(serializable_block)
                
                # Convert pending transactions
                for tx in status['pending_transactions']:
                    serializable_tx = {
                        'transaction_id': tx['transaction_id'],
                        'customer_id': tx['customer_id'],
                        'amount': tx['amount'],
                        'merchant_id': tx['merchant_id'],
                        'timestamp': tx['timestamp'],
                        'location': tx['location'],
                        'payment_method': tx['payment_method'],
                        'risk_score': tx['risk_score'],
                        'risk_level': tx['risk_level'].value if hasattr(tx['risk_level'], 'value') else str(tx['risk_level']),
                        'fraud_probability': tx['fraud_probability'],
                        'status': tx['status'].value if hasattr(tx['status'], 'value') else str(tx['status']),
                        'metadata': tx['metadata']
                    }
                    serializable_status['pending_transactions'].append(serializable_tx)
                
                return jsonify({
                    'success': True,
                    'data': serializable_status
                })
            except Exception as e:
                logger.error(f"Error getting blockchain status: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/transaction', methods=['POST'])
        def create_transaction():
            """Create and process a new transaction"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({
                        'success': False,
                        'error': 'No data provided'
                    }), 400
                
                # Validate required fields
                required_fields = ['customer_id', 'amount', 'merchant_id', 'location', 'payment_method']
                for field in required_fields:
                    if field not in data:
                        return jsonify({
                            'success': False,
                            'error': f'Missing required field: {field}'
                        }), 400
                
                # Create transaction
                transaction = self.blockchain_manager.create_transaction(
                    customer_id=data['customer_id'],
                    amount=float(data['amount']),
                    merchant_id=data['merchant_id'],
                    location=data['location'],
                    payment_method=data['payment_method'],
                    risk_score=float(data.get('risk_score', 0.5)),
                    fraud_probability=float(data.get('fraud_probability', 0.1)),
                    metadata=data.get('metadata', {})
                )
                
                # Process transaction
                result = self.blockchain_manager.process_transaction(transaction)
                
                return jsonify({
                    'success': True,
                    'data': {
                        'transaction_id': transaction.transaction_id,
                        'result': result,
                        'transaction': {
                            'transaction_id': transaction.transaction_id,
                            'customer_id': transaction.customer_id,
                            'amount': transaction.amount,
                            'merchant_id': transaction.merchant_id,
                            'timestamp': transaction.timestamp,
                            'location': transaction.location,
                            'payment_method': transaction.payment_method,
                            'risk_score': transaction.risk_score,
                            'risk_level': transaction.risk_level.value,
                            'fraud_probability': transaction.fraud_probability,
                            'status': transaction.status.value,
                            'metadata': transaction.metadata
                        }
                    }
                })
                
            except ValueError as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid data format: {str(e)}'
                }), 400
            except Exception as e:
                logger.error(f"Error creating transaction: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/transactions', methods=['GET'])
        def get_transactions():
            """Get all transactions"""
            try:
                status = self.blockchain_manager.get_blockchain_status()
                all_transactions = []
                
                # Get transactions from blocks
                for block in status['chain']:
                    for tx in block['transactions']:
                        all_transactions.append(tx)
                
                # Add pending transactions
                all_transactions.extend(status['pending_transactions'])
                
                # Sort by timestamp (newest first)
                all_transactions.sort(key=lambda x: x['timestamp'], reverse=True)
                
                return jsonify({
                    'success': True,
                    'data': {
                        'transactions': all_transactions,
                        'total_count': len(all_transactions)
                    }
                })
                
            except Exception as e:
                logger.error(f"Error getting transactions: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/transactions/<customer_id>', methods=['GET'])
        def get_customer_transactions(customer_id):
            """Get transactions for a specific customer"""
            try:
                transactions = self.blockchain_manager.get_customer_transactions(customer_id)
                
                return jsonify({
                    'success': True,
                    'data': {
                        'customer_id': customer_id,
                        'transactions': transactions,
                        'total_count': len(transactions)
                    }
                })
                
            except Exception as e:
                logger.error(f"Error getting customer transactions: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/blocks', methods=['GET'])
        def get_blocks():
            """Get all blocks"""
            try:
                status = self.blockchain_manager.get_blockchain_status()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'blocks': status['chain'],
                        'total_count': len(status['chain'])
                    }
                })
                
            except Exception as e:
                logger.error(f"Error getting blocks: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/blocks/<int:block_index>', methods=['GET'])
        def get_block(block_index):
            """Get specific block by index"""
            try:
                status = self.blockchain_manager.get_blockchain_status()
                
                if block_index >= len(status['chain']):
                    return jsonify({
                        'success': False,
                        'error': f'Block {block_index} does not exist'
                    }), 404
                
                block = status['chain'][block_index]
                
                return jsonify({
                    'success': True,
                    'data': block
                })
                
            except Exception as e:
                logger.error(f"Error getting block: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/validate', methods=['POST'])
        def validate_transaction():
            """Validate a transaction using smart contract"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({
                        'success': False,
                        'error': 'No data provided'
                    }), 400
                
                # Create transaction object for validation
                transaction = Transaction(
                    transaction_id=data.get('transaction_id', 'TEMP'),
                    customer_id=data['customer_id'],
                    amount=float(data['amount']),
                    merchant_id=data['merchant_id'],
                    timestamp=time.time(),
                    location=data['location'],
                    payment_method=data['payment_method'],
                    risk_score=float(data.get('risk_score', 0.5)),
                    risk_level=RiskLevel.SAFE,  # Will be calculated
                    fraud_probability=float(data.get('fraud_probability', 0.1)),
                    status=TransactionStatus.PENDING,
                    metadata=data.get('metadata', {})
                )
                
                # Get customer history
                customer_history = self.blockchain_manager.blockchain.get_transaction_history(data['customer_id'])
                
                # Validate using smart contract
                validation_result = self.blockchain_manager.blockchain.smart_contract.validate_transaction(
                    transaction, customer_history
                )
                
                return jsonify({
                    'success': True,
                    'data': {
                        'transaction': {
                            'transaction_id': transaction.transaction_id,
                            'customer_id': transaction.customer_id,
                            'amount': transaction.amount,
                            'merchant_id': transaction.merchant_id,
                            'timestamp': transaction.timestamp,
                            'location': transaction.location,
                            'payment_method': transaction.payment_method,
                            'risk_score': transaction.risk_score,
                            'risk_level': transaction.risk_level.value,
                            'fraud_probability': transaction.fraud_probability,
                            'status': transaction.status.value,
                            'metadata': transaction.metadata
                        },
                        'validation_result': validation_result,
                        'customer_history_count': len(customer_history)
                    }
                })
                
            except Exception as e:
                logger.error(f"Error validating transaction: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/analytics', methods=['GET'])
        def get_analytics():
            """Get fraud detection analytics"""
            try:
                analytics = self.blockchain_manager.get_fraud_analytics()
                
                return jsonify({
                    'success': True,
                    'data': analytics
                })
                
            except Exception as e:
                logger.error(f"Error getting analytics: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/mine', methods=['POST'])
        def mine_block():
            """Force mine a new block"""
            try:
                block = self.blockchain_manager.blockchain.mine_pending_transactions()
                
                if block:
                    # Convert block to serializable format
                    serializable_block = {
                        'index': block.index,
                        'timestamp': block.timestamp,
                        'transactions': [],
                        'previous_hash': block.previous_hash,
                        'nonce': block.nonce,
                        'merkle_root': block.merkle_root,
                        'block_hash': block.block_hash
                    }
                    
                    # Convert transactions
                    for tx in block.transactions:
                        serializable_tx = {
                            'transaction_id': tx.transaction_id,
                            'customer_id': tx.customer_id,
                            'amount': tx.amount,
                            'merchant_id': tx.merchant_id,
                            'timestamp': tx.timestamp,
                            'location': tx.location,
                            'payment_method': tx.payment_method,
                            'risk_score': tx.risk_score,
                            'risk_level': tx.risk_level.value,
                            'fraud_probability': tx.fraud_probability,
                            'status': tx.status.value,
                            'metadata': tx.metadata
                        }
                        serializable_block['transactions'].append(serializable_tx)
                    
                    return jsonify({
                        'success': True,
                        'data': {
                            'message': 'Block mined successfully',
                            'block': serializable_block
                        }
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No pending transactions to mine'
                    }), 400
                
            except Exception as e:
                logger.error(f"Error mining block: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/smart-contract/rules', methods=['GET'])
        def get_smart_contract_rules():
            """Get smart contract rules"""
            try:
                rules = self.blockchain_manager.blockchain.smart_contract.rules
                
                return jsonify({
                    'success': True,
                    'data': rules
                })
                
            except Exception as e:
                logger.error(f"Error getting smart contract rules: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/smart-contract/rules', methods=['PUT'])
        def update_smart_contract_rules():
            """Update smart contract rules"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({
                        'success': False,
                        'error': 'No data provided'
                    }), 400
                
                # Update rules
                for rule, value in data.items():
                    if rule in self.blockchain_manager.blockchain.smart_contract.rules:
                        self.blockchain_manager.blockchain.smart_contract.rules[rule] = value
                
                return jsonify({
                    'success': True,
                    'data': {
                        'message': 'Rules updated successfully',
                        'rules': self.blockchain_manager.blockchain.smart_contract.rules
                    }
                })
                
            except Exception as e:
                logger.error(f"Error updating smart contract rules: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/transaction/<transaction_id>', methods=['GET'])
        def get_transaction(transaction_id):
            """Get specific transaction by ID"""
            try:
                status = self.blockchain_manager.get_blockchain_status()
                
                # Search in blocks
                for block in status['chain']:
                    for tx in block['transactions']:
                        if tx['transaction_id'] == transaction_id:
                            return jsonify({
                                'success': True,
                                'data': {
                                    'transaction': tx,
                                    'block_index': block['index'],
                                    'block_hash': block['block_hash']
                                }
                            })
                
                # Search in pending transactions
                for tx in status['pending_transactions']:
                    if tx['transaction_id'] == transaction_id:
                        return jsonify({
                            'success': True,
                            'data': {
                                'transaction': tx,
                                'status': 'pending'
                            }
                        })
                
                return jsonify({
                    'success': False,
                    'error': f'Transaction {transaction_id} not found'
                }), 404
                
            except Exception as e:
                logger.error(f"Error getting transaction: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/blockchain/verify', methods=['POST'])
        def verify_transaction():
            """Verify transaction integrity"""
            try:
                data = request.get_json()
                
                if not data or 'transaction_id' not in data:
                    return jsonify({
                        'success': False,
                        'error': 'Transaction ID required'
                    }), 400
                
                transaction_id = data['transaction_id']
                status = self.blockchain_manager.get_blockchain_status()
                
                # Find transaction
                found_transaction = None
                block_index = None
                
                for i, block in enumerate(status['chain']):
                    for tx in block['transactions']:
                        if tx['transaction_id'] == transaction_id:
                            found_transaction = tx
                            block_index = i
                            break
                    if found_transaction:
                        break
                
                if not found_transaction:
                    return jsonify({
                        'success': False,
                        'error': f'Transaction {transaction_id} not found'
                    }), 404
                
                # Verify block integrity
                if block_index is not None:
                    block = status['chain'][block_index]
                    blockchain_block = self.blockchain_manager.blockchain.chain[block_index]
                    is_valid = block['block_hash'] == blockchain_block.calculate_hash()
                else:
                    is_valid = False
                
                return jsonify({
                    'success': True,
                    'data': {
                        'transaction': found_transaction,
                        'block_index': block_index,
                        'block_hash': block['block_hash'],
                        'block_valid': is_valid,
                        'chain_valid': status['is_valid']
                    }
                })
                
            except Exception as e:
                logger.error(f"Error verifying transaction: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

def create_app():
    """Create and configure Flask app"""
    api_server = BlockchainAPIServer()
    return app

if __name__ == '__main__':
    app = create_app()
    logger.info("Starting Blockchain API Server...")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  GET  /api/blockchain/status - Blockchain status")
    logger.info("  POST /api/blockchain/transaction - Create transaction")
    logger.info("  GET  /api/blockchain/transactions - Get all transactions")
    logger.info("  GET  /api/blockchain/blocks - Get all blocks")
    logger.info("  POST /api/blockchain/validate - Validate transaction")
    logger.info("  GET  /api/blockchain/analytics - Get analytics")
    logger.info("  POST /api/blockchain/mine - Mine block")
    
    app.run(host='0.0.0.0', port=5001, debug=True) 