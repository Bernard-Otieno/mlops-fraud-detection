import joblib
import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

class UnifiedFraudDetector:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.models_dir = self.base_path / 'models'
        
        # Load the full pipelines (which include the TfidfVectorizer and Scalers)
        self.transaction_model = self._load_model('fraud_detector_pipeline.joblib')
        self.promotion_model = self._load_model('promotion_fraud_model.joblib')

    def _load_model(self, filename):
        path = self.models_dir / filename
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")
            return None

    def _extract_features(self, text, sender_id):
        """
        Replicates the feature engineering from train_promotion_model.py
        """
        text_lower = text.lower()
        
        # Basic Numeric Features expected by your preprocessor
        features = {
            'sender_length': len(str(sender_id)),
            'sender_is_number': 1 if re.search(r'\d', str(sender_id)) else 0,
            'is_legit_sender': 1 if str(sender_id).upper() in ['MPESA', 'SAFARICOM'] else 0,
            'has_link': 1 if 'http' in text_lower or '.com' in text_lower or '.me' in text_lower else 0,
            'msg_length': len(text),
            'has_amount': 1 if 'ksh' in text_lower or 'sh' in text_lower else 0,
            'urgency_score': sum(1 for word in ['urgent', 'immediately', 'now', 'blocked'] if word in text_lower),
            # Mocking behavioral features used in leaderboard training
            'sender_seen_before': 0, 
            'burst_activity': 0,
            'reused_amount': 0
        }
        
        # Create a DataFrame because the Scikit-learn Pipeline expects X to be a DataFrame/Array
        df = pd.DataFrame([features])
        df['cleaned_text'] = text  # The Text feature for the TfidfVectorizer
        return df

    def predict(self, message, sender_id="UNKNOWN"):
        # 1. Prepare data in the format the model was trained on
        input_df = self._extract_features(message, sender_id)
        
        # 2. Determine which model to use (Promotion vs Transaction)
        # If it looks like a win/prize, use promotion model
        is_promo = any(word in message.lower() for word in ['won', 'prize', 'congratulations', 'gift'])
        model = self.promotion_model if (is_promo and self.promotion_model) else self.transaction_model

        if model:
            try:
                # The model is a Pipeline, so it handles scaling and TF-IDF automatically
                prob = model.predict_proba(input_df)[0][1]
            except Exception as e:
                print(f"Prediction error: {e}")
                prob = 0.5 # Fallback
        else:
            prob = 0.1 # No model loaded

        # 3. Decision Logic
        if prob > 0.85:
            risk = "🔴 CRITICAL"
            rec = "🚨 SCAM CONFIRMED. Do not click links or share your PIN."
        elif prob > 0.5:
            risk = "🟠 HIGH"
            rec = "⚠️ Highly suspicious. This matches known fraud patterns."
        elif prob > 0.25:
            risk = "🟡 MEDIUM"
            rec = "👀 Possible marketing or spam. Proceed with caution."
        else:
            risk = "🟢 LOW"
            rec = "✅ This message appears to be safe."

        return {
            "risk_level": risk,
            "fraud_probability": float(prob),
            "is_fraud": prob > 0.5,
            "recommendation": rec
        }