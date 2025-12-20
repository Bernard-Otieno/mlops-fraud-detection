"""
Quick test script for fraud prediction
"""

from src.predict_fraud import FraudDetector

# Initialize detector
detector = FraudDetector()

# Test message
message = input("\nPaste your M-PESA SMS here: ")
sender = input("Sender ID (press Enter for 'MPESA'): ").strip() or 'MPESA'

# Predict
result = detector.predict_single(message, sender)

# Display
detector.display_result(result)

# Show raw probabilities
print("Raw prediction:")
print(f"  Fraud: {result['fraud_probability']:.4f}")
print(f"  Legit: {result['legitimate_probability']:.4f}")