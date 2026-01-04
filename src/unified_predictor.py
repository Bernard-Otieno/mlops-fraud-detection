"""
Unified M-PESA Fraud Detector
Routes messages to appropriate model (Transaction or Promotion)
"""

import joblib
import pandas as pd
import numpy as np
import re
import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import feature extractors
from src.feature_extractor import extract_features as extract_transaction_features
from src.promotion_feature_extractor import extract_all_features as extract_promotion_features

class UnifiedFraudDetector:
    """
    Unified fraud detector that routes messages to the correct specialized model
    """
    
    def __init__(
        self,
        transaction_model_path='models/fraud_detector_pipeline.joblib',
        promotion_model_path='models/promotion_fraud_detector.joblib'
    ):
        """Load both specialized models"""
        
        print("="*70)
        print("🛡️ UNIFIED M-PESA FRAUD DETECTOR")
        print("="*70)
        
        # Load transaction fraud model
        if os.path.exists(transaction_model_path):
            self.transaction_model = joblib.load(transaction_model_path)
            print(f"✅ Transaction model loaded: {transaction_model_path}")
            
            # Load metadata
            trans_meta_path = transaction_model_path.replace('.joblib', '_metadata.json')
            if os.path.exists(trans_meta_path):
                with open(trans_meta_path, 'r') as f:
                    self.transaction_metadata = json.load(f)
                print(f"   Performance: F1={self.transaction_metadata['performance']['f1']:.3f}")
        else:
            self.transaction_model = None
            print(f"⚠️  Transaction model not found: {transaction_model_path}")
        
        # Load promotion fraud model
        if os.path.exists(promotion_model_path):
            self.promotion_model = joblib.load(promotion_model_path)
            print(f"✅ Promotion model loaded: {promotion_model_path}")
            
            # Load metadata
            promo_meta_path = 'models/promotion_model_metadata.json'
            if os.path.exists(promo_meta_path):
                with open(promo_meta_path, 'r') as f:
                    self.promotion_metadata = json.load(f)
                print(f"   Performance: F1={self.promotion_metadata['performance']['f1_score']:.3f}")
        else:
            self.promotion_model = None
            print(f"⚠️  Promotion model not found: {promotion_model_path}")
        
        print("="*70 + "\n")
    
    def classify_message_type(self, message_text):
        """
        Determine if message is transaction or promotion
        
        Returns:
            str: 'transaction', 'promotion', or 'unknown'
        """
        
        text_lower = message_text.lower()
        
        # Transaction indicators
        transaction_keywords = [
            'confirmed',
            'you have received',
            'paid to',
            'new m-pesa balance',
            'transaction cost'
        ]
        
        # Promotion indicators
        promotion_keywords = [
            'win', 'won', 'winner', 'congratulations',
            'prize', 'reward', 'bonus', 'offer',
            'promotion', 'competition', 'stand a chance',
            'cashback', 'discount', 'free',
            'dial *', 'sms', 'enter'
        ]
        
        # Count matches
        transaction_score = sum(1 for kw in transaction_keywords if kw in text_lower)
        promotion_score = sum(1 for kw in promotion_keywords if kw in text_lower)
        
        # Check for transaction format (amount + balance)
        has_amount = bool(re.search(r'ksh\s*\d+', text_lower))
        has_balance = 'balance' in text_lower
        
        if has_amount and has_balance and transaction_score >= 2:
            return 'transaction'
        elif promotion_score >= 2:
            return 'promotion'
        elif transaction_score > promotion_score:
            return 'transaction'
        elif promotion_score > transaction_score:
            return 'promotion'
        else:
            # Default to transaction if unclear (safer)
            return 'transaction'
    
    def predict(self, message_text, sender_id='MPESA'):
        """
        Predict fraud for any M-PESA message
        
        Args:
            message_text (str): SMS message content
            sender_id (str): Sender ID
        
        Returns:
            dict: Comprehensive prediction results
        """
        
        # Classify message type
        message_type = self.classify_message_type(message_text)
        
        # Route to appropriate model
        if message_type == 'promotion' and self.promotion_model is not None:
            return self._predict_promotion(message_text, sender_id)
        elif message_type == 'transaction' and self.transaction_model is not None:
            return self._predict_transaction(message_text, sender_id)
        else:
            # Fallback
            if self.transaction_model is not None:
                return self._predict_transaction(message_text, sender_id)
            elif self.promotion_model is not None:
                return self._predict_promotion(message_text, sender_id)
            else:
                raise ValueError("No models loaded!")
    
    def _predict_transaction(self, message_text, sender_id):
        """Predict using transaction fraud model"""
        
        # Extract features
        base_features = extract_transaction_features(message_text, sender_id)
        
        # Add behavioral features (set to 0 for single prediction)
        base_features['sender_seen_before'] = 0
        base_features['burst_activity'] = 0
        base_features['reused_amount'] = 0
        
        # Required features for transaction model
        required_features = [
            'message_length', 'action_verb_count', 'has_urgent',
            'exclamation_ratio', 'reused_amount', 'transaction_completeness',
            'is_valid_sender', 'has_link', 'link_with_urgency',
            'link_without_transaction', 'has_spelling_error',
            'soft_action_count', 'authority_count', 'sender_seen_before',
            'burst_activity'
        ]
        
        # Build feature vector
        feature_dict = {col: base_features.get(col, 0) for col in required_features}
        feature_dict['message_text'] = message_text
        
        df = pd.DataFrame([feature_dict])
        
        # Predict
        prediction = self.transaction_model.predict(df)[0]
        probability = self.transaction_model.predict_proba(df)[0]
        
        return self._format_result(
            prediction, probability, base_features,
            'transaction', message_text, sender_id
        )
    
    def _predict_promotion(self, message_text, sender_id):
        """Predict using promotion fraud model"""
        
        # Extract features
        all_features = extract_promotion_features(message_text, sender_id)
        
        # Required features for promotion model (from training)
        required_features = [
            'is_legit_sender', 'sender_is_number', 'sender_length',
            'suspicious_sender_name', 'has_link', 'link_count',
            'has_legit_domain', 'has_fraud_shortener', 'has_typosquat_domain',
            'link_at_end', 'link_at_start', 'has_ussd_code', 'ussd_code_count',
            'has_sms_shortcode', 'sms_shortcode_count', 'has_official_contact',
            'has_mobile_number', 'mobile_number_count', 'has_win_claim',
            'win_claim_count', 'has_prize_amount', 'max_prize_amount',
            'unrealistic_prize', 'extreme_prize', 'has_guarantee',
            'guarantee_count', 'has_payment_request', 'suspicious_fee_count',
            'has_suspicious_fee', 'pay_to_claim_pattern', 'urgency_count',
            'has_urgency', 'high_urgency', 'has_time_pressure',
            'mentions_expiration', 'exclamation_count', 'exclamation_density',
            'excessive_exclamations', 'question_count', 'caps_word_count',
            'excessive_caps', 'emoji_count', 'has_emoji', 'message_length',
            'word_count', 'safaricom_brand_count', 'has_safaricom_brand',
            'mentions_terms', 'says_stand_a_chance', 'says_you_won',
            'mentions_optout', 'hashtag_count', 'has_hashtag', 'says_dial',
            'has_sms_instruction', 'has_click_bait', 'requests_verification',
            'claims_authority', 'fraud_pattern_count', 'has_fraud_pattern',
            'win_fee_urgency_combo', 'link_payment_combo',
            'unrealistic_prize_fee', 'no_legit_contact_payment',
            'fraud_link_win_combo', 'legitimacy_score', 'high_legitimacy',
            'fraud_risk_score', 'high_fraud_risk'
        ]
        
        # Build feature vector
        feature_dict = {col: all_features.get(col, 0) for col in required_features}
        feature_dict['message_text'] = message_text
        
        df = pd.DataFrame([feature_dict])
        
        # Predict
        prediction = self.promotion_model.predict(df)[0]
        probability = self.promotion_model.predict_proba(df)[0]
        
        return self._format_result(
            prediction, probability, all_features,
            'promotion', message_text, sender_id
        )
    
    def _format_result(self, prediction, probability, features, 
                   message_type, message_text, sender_id):
        """Format prediction result into comprehensive dict"""    
        fraud_prob = probability[1]
        legit_prob = probability[0]
        
        # ===== ADD THIS SECTION AT THE TOP =====
        # Check if from verified financial institution
        is_verified_financial = features.get('is_verified_financial', 0)
        financial_legitimacy_score = features.get('financial_legitimacy_score', 0)
        is_legit_financial_sender = features.get('is_legit_financial_sender', 0)
        
        # Strong override for verified financial institutions
        if is_verified_financial and financial_legitimacy_score >= 4:
            # Significantly reduce fraud probability
            original_fraud_prob = fraud_prob
            fraud_prob = fraud_prob * 0.2  # Reduce to 20% of original
            legit_prob = 1 - fraud_prob
            financial_override = True
            override_reason = f"Verified financial institution (score: {financial_legitimacy_score}/8)"
        elif is_legit_financial_sender and financial_legitimacy_score >= 2:
            # Moderate reduction
            original_fraud_prob = fraud_prob
            fraud_prob = fraud_prob * 0.5  # Reduce to 50% of original
            legit_prob = 1 - fraud_prob
            financial_override = True
            override_reason = f"Likely financial institution (score: {financial_legitimacy_score}/8)"
        else:
            financial_override = False
            override_reason = None
        # ========================================
        
        # Determine risk level
        if fraud_prob >= 0.9:
            risk_level = "🔴 CRITICAL"
            recommendation = "⛔ BLOCK - Extremely high confidence fraud"
        elif fraud_prob >= 0.7:
            risk_level = "🟠 HIGH"
            recommendation = "⚠️  WARN - High likelihood of fraud"
        elif fraud_prob >= 0.5:
            risk_level = "🟡 MEDIUM"
            recommendation = "⚡ ALERT - Suspicious patterns detected"
        elif fraud_prob >= 0.3:
            risk_level = "🟢 LOW-MEDIUM"
            recommendation = "👀 REVIEW - Some suspicious elements"
        else:
            risk_level = "🟢 LOW"
            recommendation = "✅ SAFE - Appears legitimate"
        
        # Override recommendation if financial institution
        if financial_override and fraud_prob < 0.5:
            risk_level = "🟢 LOW"
            recommendation = f"✅ SAFE - {override_reason}"
        
        # ... rest of existing indicator code ...
        
        # Identify key indicators
        indicators = []
        
        # Add financial verification to indicators if present
        if is_verified_financial:
            indicators.append(f"✅ Verified financial institution (confidence: {financial_legitimacy_score}/8)")
        elif is_legit_financial_sender:
            indicators.append(f"ℹ️  Known financial sender: {sender_id}")
        
        # ... rest of existing indicator code ...
        
        # ===== UPDATE RETURN STATEMENT =====
        return {
            'is_fraud': bool(prediction) and not (financial_override and fraud_prob < 0.5),  # Override if verified
            'fraud_probability': float(fraud_prob),
            'legitimate_probability': float(legit_prob),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'message_type': message_type,
            'fraud_indicators': indicators,
            'model_used': 'promotion_model' if message_type == 'promotion' else 'transaction_model',
            'confidence': 'high' if abs(fraud_prob - 0.5) > 0.3 else 'medium' if abs(fraud_prob - 0.5) > 0.1 else 'low',
            'verified_financial': bool(is_verified_financial),
            'financial_legitimacy_score': int(financial_legitimacy_score),
            'financial_override_applied': financial_override
        }
        
    def display_result(self, result):
        """Pretty print prediction result"""
        
        print("\n" + "="*70)
        print("🎯 FRAUD DETECTION RESULT")
        print("="*70)
        
        print(f"\n📱 Message Type: {result['message_type'].upper()}")
        print(f"🤖 Model Used: {result['model_used'].replace('_', ' ').title()}")
        
        print(f"\n{result['risk_level']} RISK")
        print(f"Fraud Probability: {result['fraud_probability']*100:.1f}%")
        print(f"Legitimate Probability: {result['legitimate_probability']*100:.1f}%")
        print(f"Confidence: {result['confidence'].upper()}")
        
        print(f"\n💡 Recommendation:")
        print(f"   {result['recommendation']}")
        
        print(f"\n🚨 Fraud Indicators:")
        for indicator in result['fraud_indicators']:
            print(f"   • {indicator}")
        
        print("\n" + "="*70 + "\n")


def interactive_mode():
    """Run interactive prediction mode"""
    
    detector = UnifiedFraudDetector()
    
    print("🔍 Interactive Fraud Detection Mode")
    print("Enter any M-PESA message to check for fraud (or 'quit' to exit)\n")
    
    while True:
        print("-" * 70)
        message = input("\n📱 Enter SMS message: ").strip()
        
        if message.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not message:
            print("⚠️  Please enter a message")
            continue
        
        sender = input("📤 Enter sender ID (press Enter for 'MPESA'): ").strip()
        if not sender:
            sender = 'MPESA'
        
        # Predict
        result = detector.predict(message, sender)
        detector.display_result(result)


def test_both_models():
    """Test both transaction and promotion models"""
    
    print("\n" + "="*70)
    print("🧪 TESTING BOTH MODELS")
    print("="*70)
    
    detector = UnifiedFraudDetector()
    
    # Test cases
    test_cases = [
        # Transaction messages
        {
            'message': "Confirmed. Ksh5000.00 paid to John Mwangi. on 15/12/24 at 2.30 PM New M-PESA balance is Ksh15000.00. Transaction cost, Ksh30.00.",
            'sender': 'MPESA',
            'description': 'Legitimate Transaction',
            'expected_type': 'transaction'
        },
        {
            'message': "URGENT: Your M-PESA account will be suspended. Verify immediately at https://bit.ly/mpesa-verify123",
            'sender': 'M-PESA',
            'description': 'Transaction Fraud (Phishing)',
            'expected_type': 'transaction'
        },
        
        # Promotion messages
        {
            'message': "Congratulations!! Your first call today is FREE for 10 minutes until midday! Thank you for being part of our amazing journey #Shangwe@25",
            'sender': 'Safaricom',
            'description': 'Legitimate Promotion',
            'expected_type': 'promotion'
        },
        {
            'message': "CONGRATULATIONS!!! You have WON Ksh500,000 in the M-PESA lottery! Claim now at https://bit.ly/winner-claim. Pay Ksh1000 processing fee to 0712345678!!!",
            'sender': 'MPESA',
            'description': 'Promotion Fraud (Fake Lottery)',
            'expected_type': 'promotion'
        },
    ]
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"📝 Test Case: {test['description']}")
        print(f"📱 Message: {test['message'][:80]}...")
        print(f"📤 Sender: {test['sender']}")
        print(f"🎯 Expected Type: {test['expected_type']}")
        
        result = detector.predict(test['message'], test['sender'])
        detector.display_result(result)
        
        # Verify routing
        if result['message_type'] == test['expected_type']:
            print("✅ Correctly routed to", result['model_used'])
        else:
            print(f"⚠️  Routing mismatch: expected {test['expected_type']}, got {result['message_type']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test_both_models()
        elif sys.argv[1] == 'interactive':
            interactive_mode()
    else:
        # Default: run tests
        test_both_models()