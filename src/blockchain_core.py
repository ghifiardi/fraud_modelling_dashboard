#!/usr/bin/env python3
"""
Blockchain Core for Fraud Detection Dashboard
Implements a permissioned blockchain with smart contracts for transaction validation
"""

import hashlib
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import uuid
import threading
import queue
from enum import Enum

class TransactionStatus(Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    FRAUDULENT = "fraudulent"

class RiskLevel(Enum):
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"

@dataclass
class Transaction:
    """Represents a financial transaction with fraud detection data"""
    transaction_id: str
    customer_id: str
    amount: float
    merchant_id: str
    timestamp: float
    location: str
    payment_method: str
    risk_score: float
    risk_level: RiskLevel
    fraud_probability: float
    status: TransactionStatus
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

@dataclass
class Block:
    """Represents a block in the blockchain"""
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    nonce: int
    merkle_root: str
    block_hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate the hash of the block"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'merkle_root': self.merkle_root
        }, default=str, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int) -> None:
        """Mine the block with proof of work"""
        target = "0" * difficulty
        while self.block_hash[:difficulty] != target:
            self.nonce += 1
            self.block_hash = self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'merkle_root': self.merkle_root,
            'block_hash': self.block_hash
        }

class SmartContract:
    """Smart contract for fraud detection rules"""
    
    def __init__(self):
        self.rules = {
            'high_amount_threshold': 10000.0,
            'velocity_threshold': 5,  # transactions per hour
            'location_mismatch_penalty': 0.3,
            'night_transaction_penalty': 0.2,
            'new_merchant_penalty': 0.15,
            'international_penalty': 0.25
        }
    
    def validate_transaction(self, transaction: Transaction, customer_history: List[Transaction]) -> Dict[str, Any]:
        """Validate transaction using smart contract rules"""
        validation_result = {
            'is_valid': True,
            'risk_factors': [],
            'risk_score_adjustment': 0.0,
            'recommendation': 'allow'
        }
        
        # Rule 1: High amount check
        if transaction.amount > self.rules['high_amount_threshold']:
            validation_result['risk_factors'].append('high_amount')
            validation_result['risk_score_adjustment'] += 0.3
        
        # Rule 2: Velocity check (transactions per hour)
        recent_transactions = [
            tx for tx in customer_history 
            if tx.timestamp > transaction.timestamp - 3600  # Last hour
        ]
        if len(recent_transactions) > self.rules['velocity_threshold']:
            validation_result['risk_factors'].append('high_velocity')
            validation_result['risk_score_adjustment'] += 0.4
        
        # Rule 3: Location mismatch
        if customer_history:
            last_location = customer_history[-1].location
            if last_location != transaction.location:
                validation_result['risk_factors'].append('location_mismatch')
                validation_result['risk_score_adjustment'] += self.rules['location_mismatch_penalty']
        
        # Rule 4: Night transaction check
        hour = datetime.fromtimestamp(transaction.timestamp).hour
        if hour < 6 or hour > 23:
            validation_result['risk_factors'].append('night_transaction')
            validation_result['risk_score_adjustment'] += self.rules['night_transaction_penalty']
        
        # Rule 5: New merchant check
        merchant_history = [tx.merchant_id for tx in customer_history]
        if transaction.merchant_id not in merchant_history:
            validation_result['risk_factors'].append('new_merchant')
            validation_result['risk_score_adjustment'] += self.rules['new_merchant_penalty']
        
        # Determine final recommendation
        adjusted_risk_score = transaction.risk_score + validation_result['risk_score_adjustment']
        if adjusted_risk_score > 0.7:
            validation_result['is_valid'] = False
            validation_result['recommendation'] = 'block'
        elif adjusted_risk_score > 0.5:
            validation_result['recommendation'] = 'review'
        
        return validation_result

class Blockchain:
    """Permissioned blockchain for fraud detection"""
    
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.smart_contract = SmartContract()
        self.customer_transactions: Dict[str, List[Transaction]] = {}
        self.block_time = 10  # seconds
        self.mining_thread = None
        self.mining_queue = queue.Queue()
        self.is_mining = False
        
        # Create genesis block
        self.create_genesis_block()
        
        # Start mining thread
        self.start_mining()
    
    def create_genesis_block(self) -> None:
        """Create the first block in the chain"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0",
            nonce=0,
            merkle_root=""
        )
        genesis_block.block_hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the most recent block"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """Add a transaction to the pending pool"""
        # Validate transaction format
        if not self.is_valid_transaction_format(transaction):
            return False
        
        # Add to pending transactions
        self.pending_transactions.append(transaction)
        
        # Update customer history
        if transaction.customer_id not in self.customer_transactions:
            self.customer_transactions[transaction.customer_id] = []
        self.customer_transactions[transaction.customer_id].append(transaction)
        
        # Add to mining queue
        self.mining_queue.put(transaction)
        
        return True
    
    def is_valid_transaction_format(self, transaction: Transaction) -> bool:
        """Validate transaction format"""
        required_fields = [
            'transaction_id', 'customer_id', 'amount', 'merchant_id',
            'timestamp', 'location', 'payment_method', 'risk_score',
            'risk_level', 'fraud_probability', 'status'
        ]
        
        for field in required_fields:
            if not hasattr(transaction, field):
                return False
        
        return True
    
    def mine_pending_transactions(self) -> Optional[Block]:
        """Mine pending transactions into a new block"""
        if not self.pending_transactions:
            return None
        
        # Create new block
        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=self.get_latest_block().block_hash,
            nonce=0,
            merkle_root=self.calculate_merkle_root(self.pending_transactions)
        )
        
        # Mine the block
        block.mine_block(self.difficulty)
        
        # Add to chain
        self.chain.append(block)
        
        # Clear pending transactions
        self.pending_transactions = []
        
        return block
    
    def calculate_merkle_root(self, transactions: List[Transaction]) -> str:
        """Calculate Merkle root for transactions"""
        if not transactions:
            return hashlib.sha256("".encode()).hexdigest()
        
        # Convert transactions to strings
        tx_strings = [tx.to_json() for tx in transactions]
        
        # Build Merkle tree
        while len(tx_strings) > 1:
            if len(tx_strings) % 2 == 1:
                tx_strings.append(tx_strings[-1])
            
            new_level = []
            for i in range(0, len(tx_strings), 2):
                combined = tx_strings[i] + tx_strings[i + 1]
                new_level.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_strings = new_level
        
        return tx_strings[0]
    
    def is_chain_valid(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block hash is valid
            if current_block.block_hash != current_block.calculate_hash():
                return False
            
            # Check if previous hash is correct
            if current_block.previous_hash != previous_block.block_hash:
                return False
        
        return True
    
    def get_transaction_history(self, customer_id: str) -> List[Transaction]:
        """Get transaction history for a customer"""
        return self.customer_transactions.get(customer_id, [])
    
    def get_fraud_statistics(self) -> Dict[str, Any]:
        """Get fraud detection statistics"""
        total_transactions = 0
        fraudulent_transactions = 0
        blocked_transactions = 0
        
        for block in self.chain:
            for transaction in block.transactions:
                total_transactions += 1
                if transaction.status == TransactionStatus.FRAUDULENT:
                    fraudulent_transactions += 1
                elif transaction.status == TransactionStatus.REJECTED:
                    blocked_transactions += 1
        
        return {
            'total_transactions': total_transactions,
            'fraudulent_transactions': fraudulent_transactions,
            'blocked_transactions': blocked_transactions,
            'fraud_rate': (fraudulent_transactions / total_transactions * 100) if total_transactions > 0 else 0,
            'block_rate': (blocked_transactions / total_transactions * 100) if total_transactions > 0 else 0
        }
    
    def start_mining(self) -> None:
        """Start the mining thread"""
        self.is_mining = True
        self.mining_thread = threading.Thread(target=self._mining_worker, daemon=True)
        self.mining_thread.start()
    
    def stop_mining(self) -> None:
        """Stop the mining thread"""
        self.is_mining = False
        if self.mining_thread:
            self.mining_thread.join()
    
    def _mining_worker(self) -> None:
        """Mining worker thread"""
        while self.is_mining:
            try:
                # Wait for transactions or timeout
                transaction = self.mining_queue.get(timeout=self.block_time)
                
                # Process transaction with smart contract
                customer_history = self.get_transaction_history(transaction.customer_id)
                validation_result = self.smart_contract.validate_transaction(transaction, customer_history)
                
                # Update transaction status based on validation
                if validation_result['recommendation'] == 'block':
                    transaction.status = TransactionStatus.REJECTED
                elif validation_result['recommendation'] == 'review':
                    transaction.status = TransactionStatus.PENDING
                else:
                    transaction.status = TransactionStatus.VALIDATED
                
                # Update risk score
                transaction.risk_score += validation_result['risk_score_adjustment']
                
                # Mine block if we have enough transactions
                if len(self.pending_transactions) >= 10:
                    self.mine_pending_transactions()
                
            except queue.Empty:
                # Timeout - mine any pending transactions
                if self.pending_transactions:
                    self.mine_pending_transactions()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert blockchain to dictionary"""
        return {
            'chain': [block.to_dict() for block in self.chain],
            'pending_transactions': [tx.to_dict() for tx in self.pending_transactions],
            'difficulty': self.difficulty,
            'is_valid': self.is_chain_valid(),
            'statistics': self.get_fraud_statistics()
        }

class BlockchainManager:
    """Manager class for blockchain operations"""
    
    def __init__(self):
        self.blockchain = Blockchain()
        self.transaction_counter = 0
    
    def create_transaction(self, customer_id: str, amount: float, merchant_id: str,
                          location: str, payment_method: str, risk_score: float,
                          fraud_probability: float, metadata: Optional[Dict[str, Any]] = None) -> Transaction:
        """Create a new transaction"""
        self.transaction_counter += 1
        
        transaction = Transaction(
            transaction_id=f"TX{self.transaction_counter:08d}",
            customer_id=customer_id,
            amount=amount,
            merchant_id=merchant_id,
            timestamp=time.time(),
            location=location,
            payment_method=payment_method,
            risk_score=risk_score,
            risk_level=self._get_risk_level(risk_score),
            fraud_probability=fraud_probability,
            status=TransactionStatus.PENDING,
            metadata=metadata if metadata is not None else {}
        )
        
        return transaction
    
    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to risk level"""
        if risk_score <= 0.3:
            return RiskLevel.SAFE
        elif risk_score <= 0.5:
            return RiskLevel.LOW_RISK
        elif risk_score <= 0.7:
            return RiskLevel.MEDIUM_RISK
        else:
            return RiskLevel.HIGH_RISK
    
    def process_transaction(self, transaction: Transaction) -> Dict[str, Any]:
        """Process a transaction through the blockchain"""
        # Add to blockchain
        success = self.blockchain.add_transaction(transaction)
        
        if not success:
            return {'success': False, 'error': 'Invalid transaction format'}
        
        # Get validation result
        customer_history = self.blockchain.get_transaction_history(transaction.customer_id)
        validation_result = self.blockchain.smart_contract.validate_transaction(transaction, customer_history)
        
        return {
            'success': True,
            'transaction_id': transaction.transaction_id,
            'validation_result': validation_result,
            'blockchain_status': 'pending'
        }
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get current blockchain status"""
        return self.blockchain.to_dict()
    
    def get_customer_transactions(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get transactions for a specific customer"""
        transactions = self.blockchain.get_transaction_history(customer_id)
        return [tx.to_dict() for tx in transactions]
    
    def get_fraud_analytics(self) -> Dict[str, Any]:
        """Get fraud detection analytics"""
        stats = self.blockchain.get_fraud_statistics()
        
        # Add risk level distribution
        risk_distribution = {level.value: 0 for level in RiskLevel}
        for block in self.blockchain.chain:
            for transaction in block.transactions:
                risk_distribution[transaction.risk_level.value] += 1
        
        stats['risk_distribution'] = risk_distribution
        
        return stats 