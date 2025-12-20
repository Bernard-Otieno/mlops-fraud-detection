"""
M-PESA SMS Fraud Detection - Interactive Prediction
Load trained model and predict fraud on new messages
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys

# Import your feature extractor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features

class FraudDetector:
    """Production fraud detector with loaded model"""
    
    def __init__(self, model_path='models/fraud_detector_pipeline.joblib'):
        """Load trained model and metadata"""
        
        print("="*70)
        print("🔍 M-PESA FRAUD DETECTOR")
        print("="*70)
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train a model first!")
        
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from: {model_path}")
        
        # Load metadata
        metadata_path = model_path.replace('fraud_detector_pipeline.joblib', 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✅ Model type: {self.metadata['model_type']}")
            print(f"✅ Training date: {self.metadata['training_date']}")
            print(f"✅ Performance - F1: {self.metadata['performance']['f1']:.3f}")
        else:
            self.metadata = None
        
        print("="*70 + "\n")
    
    def predict_single(self, message_text, sender_id='MPESA'):
        """
        Predict fraud for a single message
        
        Args:
            message_text (str): The SMS message text
            sender_id (str): Sender ID (default: 'MPESA')
        
        Returns:
            dict: Prediction results with probability and risk level
        """
        
        # Extract features
        features = extract_features(message_text, sender_id)
        
        # Add behavioral features (set to 0 for single message)
        # In production, these would come from a database
        features['sender_seen_before'] = 0
        features['burst_activity'] = 0
        features['reused_amount'] = 0
        
        # Create DataFrame with required structure
        feature_cols = [
            'message_length',
            'action_verb_count',
            'has_urgent',
            'exclamation_ratio',
            'reused_amount',
            'transaction_completeness',
            'is_valid_sender',
            'has_link',
            'link_with_urgency',
            'link_without_transaction',
            'has_spelling_error',
            'soft_action_count',
            'authority_count',
            'sender_seen_before',
            'burst_activity'
        ]
        
        # Build feature vector
        feature_dict = {col: features.get(col, 0) for col in feature_cols}
        feature_dict['message_text'] = message_text
        
        df = pd.DataFrame([feature_dict])
        
        # Predict
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        
        # Determine risk level
        fraud_prob = probability[1]
        if fraud_prob >= 0.8:
            risk_level = "🔴 CRITICAL"
            recommendation = "⛔ BLOCK - High confidence fraud"
        elif fraud_prob >= 0.5:
            risk_level = "🟠 HIGH"
            recommendation = "⚠️  WARN - Likely fraud, review carefully"
        elif fraud_prob >= 0.3:
            risk_level = "🟡 MEDIUM"
            recommendation = "⚡ ALERT - Suspicious patterns detected"
        else:
            risk_level = "🟢 LOW"
            recommendation = "✅ SAFE - Appears legitimate"
        
        # Identify key fraud indicators
        indicators = []
        if not features.get('is_valid_sender', 1):
            indicators.append("❌ Invalid sender ID")
        if features.get('has_link', 0):
            indicators.append("🔗 Contains link")
        if features.get('link_without_transaction', 0):
            indicators.append("⚠️  Link without transaction details")
        if features.get('has_urgent', 0):
            indicators.append("⏰ Urgent language")
        if features.get('soft_action_count', 0) > 0:
            indicators.append("🎣 Social engineering phrases")
        if features.get('has_spelling_error', 0):
            indicators.append("📝 Spelling errors")
        if features.get('authority_count', 0) > 0:
            indicators.append("👮 Authority impersonation")
        
        return {
            'is_fraud': bool(prediction),
            'fraud_probability': float(fraud_prob),
            'legitimate_probability': float(probability[0]),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'fraud_indicators': indicators if indicators else ["None detected"],
            'features': {
                'message_length': features.get('message_length', 0),
                'has_link': bool(features.get('has_link', 0)),
                'is_valid_sender': bool(features.get('is_valid_sender', 1)),
                'transaction_completeness': features.get('transaction_completeness', 0),
                'urgent_count': features.get('has_urgent', 0)
            }
        }
    
    def display_result(self, result):
        """Pretty print prediction result"""
        
        print("\n" + "="*70)
        print("🎯 FRAUD DETECTION RESULT")
        print("="*70)
        
        print(f"\n{result['risk_level']} RISK")
        print(f"Fraud Probability: {result['fraud_probability']*100:.1f}%")
        print(f"Legitimate Probability: {result['legitimate_probability']*100:.1f}%")
        
        print(f"\n💡 Recommendation:")
        print(f"   {result['recommendation']}")
        
        print(f"\n🚨 Fraud Indicators:")
        for indicator in result['fraud_indicators']:
            print(f"   • {indicator}")
        
        print(f"\n📊 Key Features:")
        print(f"   Message Length: {result['features']['message_length']} chars")
        print(f"   Valid Sender: {result['features']['is_valid_sender']}")
        print(f"   Has Link: {result['features']['has_link']}")
        print(f"   Transaction Completeness: {result['features']['transaction_completeness']:.1%}")
        
        print("\n" + "="*70 + "\n")
    
    def predict_batch(self, messages):
        """
        Predict fraud for multiple messages
        
        Args:
            messages (list): List of tuples (message_text, sender_id)
        
        Returns:
            list: List of prediction results
        """
        results = []
        for msg, sender in messages:
            result = self.predict_single(msg, sender)
            result['message_text'] = msg
            result['sender_id'] = sender
            results.append(result)
        
        return results


def interactive_mode():
    """Run interactive prediction mode"""
    
    try:
        detector = FraudDetector()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please train a model first by running: python model_leaderboard.py")
        return
    
    print("🔍 Interactive Fraud Detection Mode")
    print("Enter SMS messages to check for fraud (or 'quit' to exit)\n")
    
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
        result = detector.predict_single(message, sender)
        detector.display_result(result)


def test_examples():
    """Test with example messages"""
    
    print("\n" + "="*70)
    print("🧪 TESTING WITH EXAMPLE MESSAGES")
    print("="*70)
    
    detector = FraudDetector()
    
    # Test cases
    test_messages = [
        # Legitimate messages
        (
            "Confirmed. Ksh5000.00 paid to John Mwangi. on 15/12/24 at 2.30 PM New M-PESA balance is Ksh15000.00. Transaction cost, Ksh30.00. Amount you can transact within the day is 500000.00.",
            "MPESA",
            "Legitimate payment"
        ),
        (
            "Confirmed. You have received Ksh2500.00 from Jane Wanjiru 0712 345 678 on 15/12/24 at 3.45 PM New M-PESA balance is Ksh12500.00.",
            "MPESA",
            "Legitimate receipt"
        ),
        
        # Fraud messages
        (
            "URGENT: Your M-PESA account will be suspended. Verify immediately at https://bit.ly/mpesa-verify123",
            "M-PESA",
            "Phishing link"
        ),
        (
            "MPESA ALERT: Your account requires verification. Send your PIN to this number within 24 hours.",
            "+254700000000",
            "PIN request"
        ),
        (
            "Confirmed. You have received Ksh15000.00 from Peter Kamau 0720 456 789 on 15/12/24 at 11.30 PM New M-PESA balance is Ksh65000.00. URGENT: This was sent by mistake. Please reverse immediately.",
            "MPESA",
            "Reversal scam"
        ),
        (
            "Confimed. Ksh10000.00 payed to Mary Njoroge. on 15/12/24 at 4.20 PM New M-PESA balanse is Ksh8000.00. Trasaction cost, Ksh55.00.",
            "MPESA",
            "Spelling errors"
        ),
    ]
    
    for message, sender, description in test_messages:
        print(f"\n{'='*70}")
        print(f"📝 Test Case: {description}")
        print(f"📱 Message: {message[:80]}...")
        print(f"📤 Sender: {sender}")
        
        result = detector.predict_single(message, sender)
        detector.display_result(result)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # Run test examples
            test_examples()
        elif sys.argv[1] == 'interactive':
            # Run interactive mode
            interactive_mode()
        else:
            print("Usage:")
            print("  python predict_fraud.py test        - Run test examples")
            print("  python predict_fraud.py interactive - Interactive mode")
    else:
        # Default: run test examples
        test_examples()