"""
FIXED: Unified Fraud Predictor
Matches exact features the trained models expect
"""

import joblib
import pandas as pd
import re
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UnifiedFraudDetector:
    def __init__(self):
        """Load trained models"""
        
        # Setup paths
        self.base_path = Path(__file__).parent.parent
        self.models_dir = self.base_path / 'models'
        
        # Load models
        self.transaction_model = self._load_model('fraud_detector_pipeline.joblib')
        self.promotion_model = self._load_model('promotion_fraud_model.joblib')
        
        if not self.transaction_model and not self.promotion_model:
            logger.error("❌ CRITICAL: No models loaded!")
    
    def _load_model(self, filename):
        """Safely load a model file"""
        path = self.models_dir / filename
        
        if not path.exists():
            logger.warning(f"⚠️ Model not found: {filename}")
            return None
        
        try:
            model = joblib.load(path)
            logger.info(f"✅ Loaded: {filename}")
            return model
        except Exception as e:
            logger.error(f"❌ Error loading {filename}: {e}")
            return None
    
    def _extract_transaction_features(self, text, sender_id):
        """
        Extract features that match EXACTLY what the transaction model expects.
        These feature names MUST match what was used during training!
        """
        
        text_lower = text.lower()
        sender_upper = str(sender_id).upper()
        
        # ========== BASIC FEATURES ==========
        
        features = {}
        
        # Message length
        features['message_length'] = len(text)
        
        # Valid sender (exact match to training)
        features['is_valid_sender'] = int(
            sender_upper in ['MPESA', 'M-PESA', 'SAFARICOM']
        )
        
        # ========== ACTION VERBS ==========
        # Count action words like: send, click, verify, confirm, etc.
        action_verbs = [
            'send', 'click', 'call', 'visit', 'confirm', 
            'verify', 'update', 'secure', 'reply'
        ]
        features['action_verb_count'] = sum(
            1 for verb in action_verbs if verb in text_lower
        )
        
        # ========== URGENCY ==========
        # Urgent words
        urgent_words = [
            'urgent', 'immediately', 'now', 'asap', 'quickly',
            'blocked', 'suspended', 'locked', 'expire'
        ]
        features['has_urgent'] = int(
            any(word in text_lower for word in urgent_words)
        )
        
        # ========== EXCLAMATIONS ==========
        exclamation_count = text.count('!')
        features['exclamation_ratio'] = (
            exclamation_count / len(text) if len(text) > 0 else 0
        )
        
        # ========== TRANSACTION COMPLETENESS ==========
        # Check if message has key transaction elements
        transaction_signals = [
            'confirmed',
            'ksh',
            'balance',
            'transaction cost',
            'new m-pesa balance'
        ]
        present = sum(1 for signal in transaction_signals if signal in text_lower)
        features['transaction_completeness'] = present / len(transaction_signals)
        
        # ========== LINK FEATURES ==========
        # Find links
        links = re.findall(r'http[s]?://[^\s]+|www\.[^\s]+', text_lower)
        has_link = len(links) > 0
        
        features['has_link'] = int(has_link)
        
        # Link with urgency (link + urgent words nearby)
        features['link_with_urgency'] = 0
        if has_link:
            for link in links:
                pos = text_lower.find(link)
                context = text_lower[max(0, pos-30):pos+30]
                if any(word in context for word in urgent_words):
                    features['link_with_urgency'] = 1
                    break
        
        # Link without transaction details
        has_transaction = ('confirmed' in text_lower) and ('ksh' in text_lower)
        features['link_without_transaction'] = int(
            has_link and not has_transaction
        )
        
        # ========== SPELLING ERRORS ==========
        spelling_errors = [
            'confimed', 'payed', 'balanse', 'ballance', 
            'trasaction', 'transction'
        ]
        features['has_spelling_error'] = int(
            any(error in text_lower for error in spelling_errors)
        )
        
        # ========== SOCIAL ENGINEERING ==========
        # "Soft" action phrases that manipulate
        soft_actions = [
            'update your details', 'verify your account',
            'confirm your information', 'secure your account',
            'click the link', 'visit the link'
        ]
        features['soft_action_count'] = sum(
            1 for phrase in soft_actions if phrase in text_lower
        )
        
        # ========== AUTHORITY CLAIMS ==========
        # Impersonating authority
        authority_phrases = [
            'safaricom', 'customer care', 'support',
            'mpesa menu', 'official', 'mpesa team'
        ]
        features['authority_count'] = sum(
            1 for phrase in authority_phrases if phrase in text_lower
        )

        
        
        return features
    
    def _classify_message_type(self, text):
        """
        Determine if this is a transaction or promotion message
        """
        
        text_lower = text.lower()
        
        # Transaction indicators
        is_transaction = (
            'confirmed' in text_lower and 
            'ksh' in text_lower and 
            'balance' in text_lower
        )
        
        # Promotion indicators
        is_promotion = any(word in text_lower for word in [
            'win', 'won', 'winner', 'prize', 'congratulations',
            'promotion', 'competition', 'reward'
        ])
        
        if is_transaction:
            return 'transaction'
        elif is_promotion:
            return 'promotion'
        else:
            # Default to transaction (safer choice)
            return 'transaction'
    
    def predict(self, message_text, sender_id='UNKNOWN'):
        """
        Main prediction method
        
        Args:
            message_text (str): The SMS to analyze
            sender_id (str): Who sent it
            
        Returns:
            dict: {risk_level, fraud_probability, recommendation}
        """
        
        # Classify message type
        msg_type = self._classify_message_type(message_text)
        
        # Extract features
        features = self._extract_transaction_features(message_text, sender_id)
        
        # Add message_text column (required by pipeline)
        features['message_text'] = message_text
        
        # Convert to DataFrame (model expects this format)
        df = pd.DataFrame([features])
        
        # Select model
        if msg_type == 'promotion' and self.promotion_model:
            model = self.promotion_model
        else:
            model = self.transaction_model
        
        if model is None:
            return {
                'risk_level': '🟡 UNKNOWN',
                'fraud_probability': 0.5,
                'recommendation': 'Bot starting up... Try again in 10 seconds.'
            }
        
        # Predict
        try:
            # Get probability of fraud (index 1)
            prob = model.predict_proba(df)[0][1]
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {
                'risk_level': '⚠️ ERROR',
                'fraud_probability': 0.5,
                'recommendation': 'Could not analyze. Please try again.'
            }
        
        # Map probability to risk level
        if prob >= 0.8:
            risk_level = '🔴 CRITICAL'
            recommendation = '🚨 SCAM DETECTED! Do NOT send money or click links. Block this number immediately.'
        elif prob >= 0.6:
            risk_level = '🟠 HIGH'
            recommendation = '⚠️ Highly suspicious. This matches known fraud patterns. Verify with official Safaricom (100/234).'
        elif prob >= 0.4:
            risk_level = '🟡 MEDIUM'
            recommendation = '⚡ Suspicious elements detected. Be cautious. Do not share personal info.'
        elif prob >= 0.2:
            risk_level = '🟢 LOW-MEDIUM'
            recommendation = '👀 Some unusual patterns. If unsure, call Safaricom 100 to verify.'
        else:
            risk_level = '🟢 LOW'
            recommendation = '✅ This message appears legitimate. Standard M-PESA transaction format.'
        
        return {
            'risk_level': risk_level,
            'fraud_probability': float(prob),
            'recommendation': recommendation
        }


# ============================================================================
# Singleton instance for production
# ============================================================================
_detector_instance = None

def get_detector():
    """
    Get or create the fraud detector instance
    (Singleton pattern - only loads models once)
    """
    global _detector_instance
    
    if _detector_instance is None:
        logger.info("🔄 Initializing Fraud Detector Models...")
        _detector_instance = UnifiedFraudDetector()
        logger.info("✅ Models loaded successfully!")
    
    return _detector_instance