# test_predictor.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unified_predictor import UnifiedFraudDetector

detector = UnifiedFraudDetector()

# Test message
message = "URGENT: Your M-PESA account will be suspended. Click https://bit.ly/verify123"
result = detector.predict(message, "MPESA-SCAM")

print(result)
# Should print: {'risk_level': '🔴 CRITICAL', 'fraud_probability': 0.92, ...}