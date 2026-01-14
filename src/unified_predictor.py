"""
FIXED: Less Strict Unified Predictor
Properly handles legitimate Safaricom messages
"""

import joblib
import pandas as pd
import re
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UnifiedFraudDetector:
    def __init__(self):
        """Load trained models"""
        
        self.base_path = Path(__file__).parent.parent
        self.models_dir = self.base_path / 'models'
        
        self.transaction_model = self._load_model('fraud_detector_pipeline.joblib')
        self.promotion_model = self._load_model('promotion_fraud_detector.joblib')
        
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
    
    def _calculate_legitimacy_score(self, text, sender_id):
        """
        Calculate how legitimate a message looks (0-10 scale)
        Higher = more legitimate
        """
        
        text_lower = text.lower()
        sender_upper = sender_id.upper()
        score = 0
        
        # ===== SENDER CHECKS (0-3 points) =====
        
        # Official sender
        if sender_upper in ['MPESA', 'M-PESA', 'SAFARICOM']:
            score += 3
        elif sender_upper in ['EQUITY', 'KCB', 'COOP', 'NCBA', 'ABSA']:
            score += 2
        
        # ===== DOMAIN CHECKS (0-2 points) =====
        
        legit_domains = [
            'safaricom.co.ke', 'mpesa.co.ke', 'm-pesa.com',
            'rebrand.ly',  # Safaricom uses this
            'g.co', 'goo.gl'  # Google shorteners used by legit services
        ]
        
        if any(domain in text_lower for domain in legit_domains):
            score += 2
        
        # ===== USSD/SMS CODES (0-2 points) =====
        
        # USSD codes like *334#, *444#
        if re.search(r'\*\d{3,4}#', text):
            score += 1
        
        # SMS short codes like 22444
        if re.search(r'\b2\d{4}\b', text):
            score += 1
        
        # ===== SAFARICOM BRANDING (0-2 points) =====
        
        safaricom_terms = [
            'bonga', 'nyakua', 'shangwe', 'mwelekoni',
            'safaricom@25', 'transforming lives',
            'lipa na m-pesa'
        ]
        
        if any(term in text_lower for term in safaricom_terms):
            score += 2
        
        # ===== TRANSACTION FORMAT (0-1 point) =====
        
        # Proper M-PESA transaction format
        has_confirmed = 'confirmed' in text_lower
        has_amount = bool(re.search(r'ksh\s*[\d,]+', text_lower))
        has_balance = 'balance' in text_lower
        
        if has_confirmed and has_amount and has_balance:
            score += 1
        
        return score
    
    def _is_obviously_legit(self, text, sender_id):
        """
        Check if message is OBVIOUSLY legitimate
        These should NEVER be flagged as fraud
        """
        
        text_lower = text.lower()
        sender_upper = sender_id.upper()
        
        # Rule 1: Official sender + official domain
        if sender_upper in ['MPESA', 'M-PESA', 'SAFARICOM']:
            if any(domain in text_lower for domain in [
                'safaricom.co.ke', 'mpesa.co.ke', 'rebrand.ly'
            ]):
                return True
        
        # Rule 2: Proper M-PESA transaction from MPESA sender
        if sender_upper in ['MPESA', 'M-PESA']:
            is_transaction = (
                'confirmed' in text_lower and
                bool(re.search(r'ksh\s*[\d,]+', text_lower)) and
                'balance' in text_lower
            )
            if is_transaction:
                return True
        
        # Rule 3: Contains USSD code + Safaricom branding
        has_ussd = bool(re.search(r'\*\d{3,4}#', text))
        has_brand = any(x in text_lower for x in ['bonga', 'nyakua', 'shangwe'])
        
        if has_ussd and has_brand:
            return True
        
        # Rule 4: "Stand a chance" (legitimate promo language)
        if sender_upper in ['MPESA', 'SAFARICOM']:
            if 'stand a chance' in text_lower:
                return True
        
        return False
    
    def _extract_transaction_features(self, text, sender_id):
        """Extract features for transaction messages"""
        
        text_lower = text.lower()
        sender_upper = str(sender_id).upper()
        
        features = {}
        
        # Basic features
        features['message_length'] = len(text)
        features['is_valid_sender'] = int(
            sender_upper in ['MPESA', 'M-PESA', 'SAFARICOM']
        )
        
        # Action verbs
        action_verbs = [
            'send', 'click', 'call', 'visit', 'confirm',
            'verify', 'update', 'secure', 'reply'
        ]
        features['action_verb_count'] = sum(
            1 for verb in action_verbs if verb in text_lower
        )
        
        # Urgency
        urgent_words = [
            'urgent', 'immediately', 'now', 'asap', 'quickly',
            'blocked', 'suspended', 'locked', 'expire'
        ]
        features['has_urgent'] = int(
            any(word in text_lower for word in urgent_words)
        )
        
        # Exclamations
        exclamation_count = text.count('!')
        features['exclamation_ratio'] = (
            exclamation_count / len(text) if len(text) > 0 else 0
        )
        
        # Transaction completeness
        transaction_signals = [
            'confirmed', 'ksh', 'balance',
            'transaction cost', 'new m-pesa balance'
        ]
        present = sum(1 for signal in transaction_signals if signal in text_lower)
        features['transaction_completeness'] = present / len(transaction_signals)
        
        # Link features
        links = re.findall(r'http[s]?://[^\s]+|www\.[^\s]+', text_lower)
        has_link = len(links) > 0
        features['has_link'] = int(has_link)
        
        features['link_with_urgency'] = 0
        if has_link:
            for link in links:
                pos = text_lower.find(link)
                context = text_lower[max(0, pos-30):pos+30]
                if any(word in context for word in urgent_words):
                    features['link_with_urgency'] = 1
                    break
        
        has_transaction = ('confirmed' in text_lower) and ('ksh' in text_lower)
        features['link_without_transaction'] = int(
            has_link and not has_transaction
        )
        
        # Spelling errors
        spelling_errors = [
            'confimed', 'payed', 'balanse', 'ballance',
            'trasaction', 'transction'
        ]
        features['has_spelling_error'] = int(
            any(error in text_lower for error in spelling_errors)
        )
        
        # Social engineering
        soft_actions = [
            'update your details', 'verify your account',
            'confirm your information', 'secure your account',
            'click the link', 'visit the link'
        ]
        features['soft_action_count'] = sum(
            1 for phrase in soft_actions if phrase in text_lower
        )
        
        # Authority claims
        authority_phrases = [
            'safaricom', 'customer care', 'support',
            'mpesa menu', 'official', 'mpesa team'
        ]
        features['authority_count'] = sum(
            1 for phrase in authority_phrases if phrase in text_lower
        )
        
        return features
    
    def _classify_message_type(self, text):
        """Determine if transaction or promotion"""
        
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
            'promotion', 'competition', 'reward', 'offer', 'bonus'
        ])
        
        if is_transaction:
            return 'transaction'
        elif is_promotion:
            return 'promotion'
        else:
            return 'transaction'
    
    def predict(self, message_text, sender_id='UNKNOWN'):
        """
        Main prediction method with smart legitimacy handling
        """
        
        # ========== STEP 1: CHECK IF OBVIOUSLY LEGIT ==========
        
        if self._is_obviously_legit(message_text, sender_id):
            logger.info("✅ Message passed obvious legitimacy check")
            return {
                'risk_level': '🟢 LOW',
                'fraud_probability': 0.02,
                'recommendation': '✅ This appears to be a legitimate Safaricom/M-PESA message.'
            }
        
        # ========== STEP 2: CALCULATE LEGITIMACY SCORE ==========
        
        legitimacy_score = self._calculate_legitimacy_score(message_text, sender_id)
        logger.info(f"📊 Legitimacy score: {legitimacy_score}/10")
        
        # ========== STEP 3: EXTRACT FEATURES & CLASSIFY ==========
        
        msg_type = self._classify_message_type(message_text)
        features = self._extract_transaction_features(message_text, sender_id)
        features['message_text'] = message_text
        
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
        
        # ========== STEP 4: RUN MODEL PREDICTION ==========
        
        try:
            raw_prob = model.predict_proba(df)[0][1]
            logger.info(f"🤖 Raw model prediction: {raw_prob:.3f}")
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {
                'risk_level': '⚠️ ERROR',
                'fraud_probability': 0.5,
                'recommendation': 'Could not analyze. Please try again.'
            }
        
        # ========== STEP 5: ADJUST BASED ON LEGITIMACY SCORE ==========
        
        # Strong legitimacy signals reduce fraud probability
        if legitimacy_score >= 7:
            # Very strong legitimacy
            adjusted_prob = raw_prob * 0.2  # Reduce to 20%
            logger.info(f"📉 Strong legitimacy detected (score {legitimacy_score}), adjusted: {adjusted_prob:.3f}")
        
        elif legitimacy_score >= 5:
            # Moderate legitimacy
            adjusted_prob = raw_prob * 0.5  # Reduce to 50%
            logger.info(f"📉 Moderate legitimacy detected (score {legitimacy_score}), adjusted: {adjusted_prob:.3f}")
        
        elif legitimacy_score >= 3:
            # Some legitimacy
            adjusted_prob = raw_prob * 0.7  # Reduce to 70%
            logger.info(f"📉 Some legitimacy detected (score {legitimacy_score}), adjusted: {adjusted_prob:.3f}")
        
        else:
            # Low legitimacy - trust the model
            adjusted_prob = raw_prob
        
        # Final capping for official senders
        sender_upper = sender_id.upper()
        if sender_upper in ['MPESA', 'M-PESA', 'SAFARICOM']:
            # Official senders should NEVER exceed 70% fraud probability
            adjusted_prob = min(adjusted_prob, 0.70)
            logger.info(f"🛡️ Official sender cap applied: {adjusted_prob:.3f}")
        
        prob = adjusted_prob
        
        # ========== STEP 6: MAP TO RISK LEVEL ==========
        
        if prob >= 0.85:
            risk_level = '🔴 CRITICAL'
            recommendation = '🚨 SCAM DETECTED! Do NOT send money or click links. Block this sender immediately.'
        elif prob >= 0.70:
            risk_level = '🟠 HIGH'
            recommendation = '⚠️ Highly suspicious. Verify with official Safaricom (100/234) before taking action.'
        elif prob >= 0.50:
            risk_level = '🟡 MEDIUM'
            recommendation = '⚡ Suspicious elements detected. Be cautious. Do not share personal info.'
        elif prob >= 0.30:
            risk_level = '🟢 LOW-MEDIUM'
            recommendation = '👀 Some unusual patterns. If unsure, call Safaricom 100 to verify.'
        else:
            risk_level = '🟢 LOW'
            recommendation = '✅ This message appears legitimate based on standard patterns.'
        
        return {
            'risk_level': risk_level,
            'fraud_probability': float(prob),
            'recommendation': recommendation,
            'legitimacy_score': legitimacy_score,  # For debugging
            'raw_probability': float(raw_prob)  # For debugging
        }


# ============================================================================
# Singleton instance
# ============================================================================
_detector_instance = None

def get_detector():
    """Get or create detector instance"""
    global _detector_instance
    
    if _detector_instance is None:
        logger.info("🔄 Initializing Fraud Detector Models...")
        _detector_instance = UnifiedFraudDetector()
        logger.info("✅ Models loaded successfully!")
    
    return _detector_instance