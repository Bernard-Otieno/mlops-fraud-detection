import joblib
import os
import pandas as pd
import numpy as np
from pathlib import Path

class UnifiedFraudDetector:
    def __init__(self):
        # 1. Setup Paths
        # This assumes models are in the 'models' folder at the project root
        self.base_path = Path(__file__).parent.parent
        self.models_dir = self.base_path / 'models'
        
        # 2. Load Models
        self.transaction_model = self._load_model('transaction_fraud_model.joblib')
        self.promotion_model = self._load_model('promotion_fraud_model.joblib')

    def _load_model(self, filename):
        path = self.models_dir / filename
        try:
            return joblib.load(path)
        except FileNotFoundError:
            print(f"⚠️ Model not found: {path}")
            return None

    def predict(self, message, sender_id="UNKNOWN"):
        """
        Main prediction function called by the bot
        """
        # Default Safe Result
        result = {
            "risk_level": "LOW",
            "fraud_probability": 0.0,
            "is_fraud": False,
            "reasoning": "Analysis inconclusive",
            "recommendation": "Stay vigilant.",
            "fraud_indicators": []
        }

        # 1. Rule-Based Checks (Fastest)
        msg_lower = message.lower()
        
        # Sender Check
        if sender_id.upper() not in ['MPESA', 'SAFARICOM'] and "mpesa" in msg_lower:
            result['fraud_indicators'].append("Unofficial sender using M-PESA name")
            result['risk_level'] = "HIGH"
            result['fraud_probability'] = 0.85
        
        # Keyword Check
        suspicious_words = ['congratulations', 'won', 'redeem', 'locked', 'suspended', 'pin']
        found_words = [w for w in suspicious_words if w in msg_lower]
        if found_words:
            result['fraud_indicators'].append(f"Suspicious words: {', '.join(found_words)}")
            result['fraud_probability'] += 0.1

        # 2. ML Model Check (If models exist)
        # (Simplified for stability - relies heavily on rules if ML fails)
        if self.promotion_model and ("win" in msg_lower or "prize" in msg_lower):
            # In a real app, you would transform the text here
            pass 

        # 3. Finalize Risk Level
        prob = min(result['fraud_probability'], 1.0)
        result['fraud_probability'] = prob
        
        if prob > 0.8:
            result['risk_level'] = "🔴 CRITICAL"
            result['is_fraud'] = True
            result['recommendation'] = "DO NOT REPLY. This is a confirmed scam pattern."
        elif prob > 0.5:
            result['risk_level'] = "🟠 HIGH"
            result['is_fraud'] = True
            result['recommendation'] = "Highly suspicious. Verify with Safaricom (Call 100)."
            
        return result