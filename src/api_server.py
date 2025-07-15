#!/usr/bin/env python3
"""
FastAPI Server for AI Fraud Detection Monitoring
Provides REST API endpoints for real-time monitoring and model management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
import datetime
import asyncio
import logging
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bank_fraud_detector import BankFraudDetector
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: int
    amount: float
    transaction_type: str
    merchant_category: str
    hour: int
    day_of_week: int
    location: str
    device_type: str
    card_present: bool
    previous_fraud_flag: bool
    account_age_days: int
    balance_before: float
    balance_after: float

class TransactionResponse(BaseModel):
    transaction_id: str
    risk_level: str
    risk_probability: float
    recommended_action: str
    model_predictions: Dict[str, Dict[str, Any]]
    timestamp: str

class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    latency_ms: float

class SystemHealth(BaseModel):
    status: str
    uptime_seconds: float
    total_transactions: int
    fraud_detected: int
    success_rate: float
    avg_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float

class FraudDetectionAPI:
    def __init__(self):
        self.app = FastAPI(
            title="AI Fraud Detection API",
            description="Real-time fraud detection and monitoring API",
            version="1.0.0"
        )
        self.detector = None
        self.model_path = "models/bank_fraud_detector.pkl"
        self.transaction_history = []
        self.system_metrics = {
            'start_time': datetime.datetime.now(),
            'total_transactions': 0,
            'fraud_detected': 0,
            'response_times': []
        }
        
        self.setup_middleware()
        self.setup_routes()
        self.load_model()
    
    def setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "AI Fraud Detection API",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """System health check endpoint."""
            return self.get_system_health()
        
        @self.app.post("/predict", response_model=TransactionResponse)
        async def predict_transaction(transaction: TransactionRequest):
            """Predict fraud risk for a single transaction."""
            return await self.process_transaction(transaction)
        
        @self.app.post("/predict/batch")
        async def predict_batch(transactions: List[TransactionRequest]):
            """Predict fraud risk for multiple transactions."""
            results = []
            for transaction in transactions:
                result = await self.process_transaction(transaction)
                results.append(result)
            return {"predictions": results}
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get system metrics and performance data."""
            return self.get_system_metrics()
        
        @self.app.get("/models")
        async def get_models():
            """Get information about loaded models."""
            return self.get_model_info()
        
        @self.app.post("/models/retrain")
        async def retrain_models(background_tasks: BackgroundTasks):
            """Retrain models in the background."""
            background_tasks.add_task(self.retrain_models_background)
            return {"message": "Model retraining initiated", "status": "processing"}
        
        @self.app.get("/transactions/recent")
        async def get_recent_transactions(limit: int = 100):
            """Get recent transaction history."""
            return {"transactions": self.transaction_history[-limit:]}
        
        @self.app.get("/analytics/hourly")
        async def get_hourly_analytics():
            """Get hourly transaction analytics."""
            return self.get_hourly_analytics()
        
        @self.app.get("/analytics/daily")
        async def get_daily_analytics():
            """Get daily transaction analytics."""
            return self.get_daily_analytics()
        
        @self.app.get("/alerts")
        async def get_alerts():
            """Get recent system alerts."""
            return self.get_recent_alerts()
    
    def load_model(self):
        """Load the trained fraud detection model."""
        try:
            if os.path.exists(self.model_path):
                self.detector = BankFraudDetector()
                self.detector.load_bank_model(self.model_path)
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning("Model not found. Please train the model first.")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    async def process_transaction(self, transaction: TransactionRequest) -> TransactionResponse:
        """Process a single transaction and return prediction results."""
        start_time = datetime.datetime.now()
        
        try:
            if not self.detector:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Convert to DataFrame
            transaction_data = pd.DataFrame([transaction.dict()])
            
            # Engineer features
            transaction_data = self.detector.engineer_bank_features(transaction_data)
            
            # Get prediction
            result = self.detector.predict_transaction_risk(transaction_data.iloc[0])
            
            if not result:
                raise HTTPException(status_code=500, detail="Prediction failed")
            
            # Calculate response time
            response_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
            
            # Update system metrics
            self.system_metrics['total_transactions'] += 1
            self.system_metrics['response_times'].append(response_time)
            if result['risk_level'] in ['HIGH_RISK', 'MEDIUM_RISK']:
                self.system_metrics['fraud_detected'] += 1
            
            # Store transaction history
            transaction_record = {
                'transaction_id': transaction.transaction_id,
                'customer_id': transaction.customer_id,
                'amount': transaction.amount,
                'risk_level': result['risk_level'],
                'risk_probability': result['risk_probability'],
                'timestamp': datetime.datetime.now().isoformat(),
                'response_time_ms': response_time
            }
            self.transaction_history.append(transaction_record)
            
            return TransactionResponse(
                transaction_id=transaction.transaction_id,
                risk_level=result['risk_level'],
                risk_probability=result['risk_probability'],
                recommended_action=result['recommended_action'],
                model_predictions=result['model_predictions'],
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_system_health(self) -> SystemHealth:
        """Get system health metrics."""
        uptime = (datetime.datetime.now() - self.system_metrics['start_time']).total_seconds()
        
        # Calculate success rate
        total = self.system_metrics['total_transactions']
        success_rate = 1.0 if total == 0 else (total - len([t for t in self.transaction_history if t.get('error')])) / total
        
        # Calculate average response time
        response_times = self.system_metrics['response_times']
        avg_response_time = np.mean(response_times) if response_times else 0
        
        return SystemHealth(
            status="healthy" if self.detector else "unhealthy",
            uptime_seconds=uptime,
            total_transactions=self.system_metrics['total_transactions'],
            fraud_detected=self.system_metrics['fraud_detected'],
            success_rate=success_rate,
            avg_response_time_ms=avg_response_time,
            memory_usage_mb=self.get_memory_usage(),
            cpu_usage_percent=self.get_cpu_usage()
        )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        health = self.get_system_health()
        
        # Calculate additional metrics
        recent_transactions = self.transaction_history[-100:] if self.transaction_history else []
        
        hourly_stats = {}
        for transaction in recent_transactions:
            hour = datetime.datetime.fromisoformat(transaction['timestamp']).hour
            if hour not in hourly_stats:
                hourly_stats[hour] = {'count': 0, 'fraud_count': 0}
            hourly_stats[hour]['count'] += 1
            if transaction['risk_level'] in ['HIGH_RISK', 'MEDIUM_RISK']:
                hourly_stats[hour]['fraud_count'] += 1
        
        return {
            "health": health.dict(),
            "hourly_stats": hourly_stats,
            "risk_distribution": self.get_risk_distribution(),
            "model_performance": self.get_model_performance(),
            "recent_alerts": self.get_recent_alerts()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        if not self.detector:
            return {"error": "No models loaded"}
        
        model_info = {
            "models": {},
            "features": self.detector.feature_columns,
            "risk_thresholds": self.detector.risk_thresholds
        }
        
        for name, model in self.detector.models.items():
            model_info["models"][name] = {
                "type": type(model).__name__,
                "parameters": self.get_model_parameters(model)
            }
        
        return model_info
    
    def get_model_parameters(self, model) -> Dict[str, Any]:
        """Extract model parameters."""
        params = {}
        if hasattr(model, 'n_estimators'):
            params['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            params['max_depth'] = model.max_depth
        if hasattr(model, 'random_state'):
            params['random_state'] = model.random_state
        return params
    
    def get_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of risk levels."""
        if not self.transaction_history:
            return {}
        
        distribution = {}
        for transaction in self.transaction_history:
            risk_level = transaction['risk_level']
            distribution[risk_level] = distribution.get(risk_level, 0) + 1
        
        return distribution
    
    def get_model_performance(self) -> List[ModelMetrics]:
        """Get model performance metrics."""
        # This would typically come from actual model evaluation
        # For now, return sample metrics
        return [
            ModelMetrics(
                model_name="Logistic Regression",
                accuracy=0.85,
                precision=0.82,
                recall=0.78,
                f1_score=0.80,
                auc_score=0.85,
                latency_ms=50
            ),
            ModelMetrics(
                model_name="Random Forest",
                accuracy=0.92,
                precision=0.89,
                recall=0.85,
                f1_score=0.87,
                auc_score=0.92,
                latency_ms=120
            ),
            ModelMetrics(
                model_name="Isolation Forest",
                accuracy=0.78,
                precision=0.75,
                recall=0.72,
                f1_score=0.73,
                auc_score=0.78,
                latency_ms=80
            )
        ]
    
    def get_hourly_analytics(self) -> Dict[str, Any]:
        """Get hourly transaction analytics."""
        if not self.transaction_history:
            return {}
        
        hourly_data = {}
        for transaction in self.transaction_history:
            hour = datetime.datetime.fromisoformat(transaction['timestamp']).hour
            if hour not in hourly_data:
                hourly_data[hour] = {
                    'count': 0,
                    'total_amount': 0,
                    'fraud_count': 0,
                    'avg_response_time': 0,
                    'response_times': []
                }
            
            hourly_data[hour]['count'] += 1
            hourly_data[hour]['total_amount'] += transaction['amount']
            hourly_data[hour]['response_times'].append(transaction['response_time_ms'])
            
            if transaction['risk_level'] in ['HIGH_RISK', 'MEDIUM_RISK']:
                hourly_data[hour]['fraud_count'] += 1
        
        # Calculate averages
        for hour in hourly_data:
            response_times = hourly_data[hour]['response_times']
            hourly_data[hour]['avg_response_time'] = np.mean(response_times) if response_times else 0
            hourly_data[hour]['avg_amount'] = hourly_data[hour]['total_amount'] / hourly_data[hour]['count']
            del hourly_data[hour]['response_times']
        
        return hourly_data
    
    def get_daily_analytics(self) -> Dict[str, Any]:
        """Get daily transaction analytics."""
        if not self.transaction_history:
            return {}
        
        daily_data = {}
        for transaction in self.transaction_history:
            date = datetime.datetime.fromisoformat(transaction['timestamp']).date().isoformat()
            if date not in daily_data:
                daily_data[date] = {
                    'count': 0,
                    'total_amount': 0,
                    'fraud_count': 0,
                    'avg_response_time': 0,
                    'response_times': []
                }
            
            daily_data[date]['count'] += 1
            daily_data[date]['total_amount'] += transaction['amount']
            daily_data[date]['response_times'].append(transaction['response_time_ms'])
            
            if transaction['risk_level'] in ['HIGH_RISK', 'MEDIUM_RISK']:
                daily_data[date]['fraud_count'] += 1
        
        # Calculate averages
        for date in daily_data:
            response_times = daily_data[date]['response_times']
            daily_data[date]['avg_response_time'] = np.mean(response_times) if response_times else 0
            daily_data[date]['avg_amount'] = daily_data[date]['total_amount'] / daily_data[date]['count']
            del daily_data[date]['response_times']
        
        return daily_data
    
    def get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent system alerts."""
        alerts = []
        
        # Check for high-risk transactions
        recent_high_risk = [t for t in self.transaction_history[-50:] if t['risk_level'] == 'HIGH_RISK']
        if recent_high_risk:
            alerts.append({
                "level": "HIGH",
                "message": f"High-risk transaction detected: {recent_high_risk[-1]['transaction_id']}",
                "timestamp": recent_high_risk[-1]['timestamp'],
                "type": "fraud_detection"
            })
        
        # Check for system performance
        recent_response_times = [t['response_time_ms'] for t in self.transaction_history[-10:]]
        if recent_response_times and np.mean(recent_response_times) > 1000:
            alerts.append({
                "level": "MEDIUM",
                "message": "High response time detected",
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "performance"
            })
        
        # Check for model health
        if not self.detector:
            alerts.append({
                "level": "HIGH",
                "message": "No models loaded",
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "system"
            })
        
        return alerts
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    async def retrain_models_background(self):
        """Background task to retrain models."""
        logger.info("Starting model retraining...")
        # Add retraining logic here
        await asyncio.sleep(10)  # Simulate training time
        logger.info("Model retraining completed")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    api = FraudDetectionAPI()
    return api.app

# For running with uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 