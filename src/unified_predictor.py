import joblib
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

class UnifiedFraudDetector:
    def __init__(self):
        # Setup paths relative to this file
        self.base_path = Path(__file__).parent.parent
        self.models_dir = self.base_path / 'models'
        
        # Load the saved pipelines (which include the TfidfVectorizer)
        self.transaction_model = self._load_model('fraud_detector_pipeline.joblib')
        self.promotion_model = self._load_model('promotion_fraud_model.joblib')
        
        if not self.transaction_model and not self.promotion_model:
            print("❌ CRITICAL: No models were loaded. Accuracy will be 10%.")

    def _load_model(self, filename):
        path = self.models_dir / filename
        if not path.exists():
            print(f"⚠️ Model file missing at: {path}")
            return None
        try:
            model = joblib.load(path)
            print(f"✅ Loaded: {filename}")
            return model
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None

    def _extract_features(self, text, sender_id):
        """
        Exhaustive feature extraction to match NUMERIC_FEATURES in train_promotion_model.py
        """
        text_lower = text.lower()
        sender_str = str(sender_id).upper()
        
        # Links and Domains logic
        has_link = 1 if any(x in text_lower for x in ['http', '.com', '.me', 'bit.ly']) else 0
        
        features = {
            # Sender features
            'is_legit_sender': 1 if sender_str in ['MPESA', 'SAFARICOM', 'M-PESA'] else 0,
            'sender_is_number': 1 if re.search(r'\d', sender_str) else 0,
            'sender_length': len(sender_str),
            'suspicious_sender_name': 1 if any(x in sender_str for x in ['AGENT', 'OFFICE', 'WINNER', 'PRIZE']) else 0,
            
            # Link features
            'has_link': has_link,
            'link_count': text_lower.count('http') + text_lower.count('www'),
            'has_legit_domain': 1 if any(x in text_lower for x in ['safaricom.co.ke', 'm-pesa.com']) else 0,
            'has_fraud_shortener': 1 if any(x in text_lower for x in ['bit.ly', 't.co', 'tinyurl']) else 0,
            'has_typosquat_domain': 0, # Placeholder
            'link_at_end': 1 if text_lower.strip().endswith(('.com', '.me', '/')) else 0,
            'link_at_start': 0,
            
            # Content features
            'has_amount': 1 if any(x in text_lower for x in ['ksh', 'sh', 'amount']) else 0,
            'urgency_score': sum(1 for word in ['urgent', 'immediately', 'now', 'blocked', 'limit'] if word in text_lower),
            'msg_length': len(text),
            'is_all_caps': 1 if text.isupper() else 0,
            
            # Behavioral mocks (Matches model_leaderboard.py requirements)
            'sender_seen_before': 0,
            'burst_activity': 0,
            'reused_amount': 0
        }
        
        df = pd.DataFrame([features])
        df['cleaned_text'] = text # Matches TEXT_FEATURE in training script
        return df

    def predict(self, message, sender_id="UNKNOWN"):
        # 1. Feature Engineering
        input_df = self._extract_features(message, sender_id)
        
        # 2. Model Selection
        # Use promotion model if keywords exist, otherwise transaction model
        is_promo = any(w in message.lower() for w in ['won', 'congratulations', 'gift', 'prize'])
        model = self.promotion_model if (is_promo and self.promotion_model) else self.transaction_model

        if model is None:
            # If model loading failed, fallback to 10%
            return {"risk_level": "LOW", "fraud_probability": 0.1, "recommendation": "Bot starting up..."}

        try:
            # predict_proba returns [[prob_legit, prob_fraud]]
            # We want prob_fraud (index 1)
            prob = model.predict_proba(input_df)[0][1]
        except Exception as e:
            print(f"Prediction error: {e}")
            prob = 0.1

        # 3. Risk Mapping
        if prob > 0.8:
            risk, rec = "🔴 CRITICAL", "🚨 SCAM DETECTED. Do not reply or send money."
        elif prob > 0.5:
            risk, rec = "🟠 HIGH", "⚠️ Highly suspicious. Matches fraud patterns."
        elif prob > 0.3:
            risk, rec = "🟡 MEDIUM", "👀 Caution advised. Likely spam or marketing."
        else:
            risk, rec = "🟢 LOW", "✅ This message appears safe."

        return {
            "risk_level": risk,
            "fraud_probability": float(prob),
            "recommendation": rec
        }