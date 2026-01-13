"""
Unified M-PESA Fraud Detector
Routes messages to appropriate model (Transaction or Promotion)
"""

import joblib
import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Ensure project root is on path
project_root = str(BASE_DIR)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Feature extractors
from src.feature_extractor import extract_features as extract_transaction_features
from src.promotion_feature_extractor import extract_all_features as extract_promotion_features


class UnifiedFraudDetector:
    """
    Unified fraud detector that routes messages to the correct specialized model
    """

    def __init__(
        self,
        transaction_model_path=MODEL_DIR / "fraud_detector_pipeline.joblib",
        promotion_model_path=MODEL_DIR / "promotion_fraud_detector.joblib",
    ):
        print("=" * 70)
        print("🛡️ UNIFIED M-PESA FRAUD DETECTOR")
        print("=" * 70)

        # Load transaction model
        if transaction_model_path.exists():
            self.transaction_model = joblib.load(transaction_model_path)
            print(f"✅ Transaction model loaded: {transaction_model_path}")
        else:
            self.transaction_model = None
            print(f"⚠️ Transaction model not found: {transaction_model_path}")

        # Load promotion model
        if promotion_model_path.exists():
            self.promotion_model = joblib.load(promotion_model_path)
            print(f"✅ Promotion model loaded: {promotion_model_path}")
        else:
            self.promotion_model = None
            print(f"⚠️ Promotion model not found: {promotion_model_path}")

        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Message classification
    # ------------------------------------------------------------------
    def classify_message_type(self, message_text: str) -> str:
        text = message_text.lower()

        transaction_keywords = [
            "confirmed",
            "you have received",
            "paid to",
            "new m-pesa balance",
            "transaction cost",
        ]

        promotion_keywords = [
            "win",
            "won",
            "winner",
            "congratulations",
            "prize",
            "reward",
            "bonus",
            "offer",
            "promotion",
            "competition",
            "stand a chance",
            "cashback",
            "discount",
            "free",
            "dial *",
            "sms",
            "enter",
        ]

        transaction_score = sum(k in text for k in transaction_keywords)
        promotion_score = sum(k in text for k in promotion_keywords)

        has_amount = bool(re.search(r"ksh\s*\d+", text))
        has_balance = "balance" in text

        if has_amount and has_balance and transaction_score >= 2:
            return "transaction"
        if promotion_score >= 2:
            return "promotion"
        if transaction_score > promotion_score:
            return "transaction"
        if promotion_score > transaction_score:
            return "promotion"

        return "transaction"  # safe default

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------
    def predict(self, message_text: str, sender_id: str = "MPESA") -> dict:
        message_type = self.classify_message_type(message_text)

        if message_type == "promotion" and self.promotion_model:
            return self._predict_promotion(message_text, sender_id)

        if message_type == "transaction" and self.transaction_model:
            return self._predict_transaction(message_text, sender_id)

        if self.transaction_model:
            return self._predict_transaction(message_text, sender_id)

        if self.promotion_model:
            return self._predict_promotion(message_text, sender_id)

        raise RuntimeError("❌ No models loaded")

    # ------------------------------------------------------------------
    # Transaction prediction
    # ------------------------------------------------------------------
    def _predict_transaction(self, message_text, sender_id):
        base_features = extract_transaction_features(message_text, sender_id)

        # Behavioural defaults (single-message inference)
        base_features.update(
            {
                "sender_seen_before": 0,
                "burst_activity": 0,
                "reused_amount": 0,
            }
        )

        required_features = [
            "message_length",
            "action_verb_count",
            "urgent_density",
            "transaction_completeness",
            "is_valid_sender",
            "has_link",
            "link_with_urgency",
            "link_without_transaction",
            "has_spelling_error",
            "all_caps_ratio",
            "exclamation_ratio",
            "sender_seen_before",
            "burst_activity",
            "reused_amount",
        ]

        missing = [f for f in required_features if f not in base_features]
        if missing:
            print(f"⚠️ Missing transaction features: {missing}")

        feature_dict = {f: base_features.get(f, 0) for f in required_features}
        feature_dict["message_text"] = message_text

        df = pd.DataFrame([feature_dict])

        pred = self.transaction_model.predict(df)[0]
        prob = self.transaction_model.predict_proba(df)[0]

        return self._format_result(
            pred,
            prob,
            base_features,
            "transaction",
            message_text,
            sender_id,
        )

    # ------------------------------------------------------------------
    # Promotion prediction
    # ------------------------------------------------------------------
    def _predict_promotion(self, message_text, sender_id):
        all_features = extract_promotion_features(message_text, sender_id)

        required_features = [
            k for k in all_features.keys() if k != "message_text"
        ]

        feature_dict = {f: all_features.get(f, 0) for f in required_features}
        feature_dict["message_text"] = message_text

        df = pd.DataFrame([feature_dict])

        pred = self.promotion_model.predict(df)[0]
        prob = self.promotion_model.predict_proba(df)[0]

        return self._format_result(
            pred,
            prob,
            all_features,
            "promotion",
            message_text,
            sender_id,
        )

    # ------------------------------------------------------------------
    # Result formatting (SAFE VERSION)
    # ------------------------------------------------------------------
    def _format_result(
        self,
        prediction,
        probability,
        features,
        message_type,
        message_text,
        sender_id,
    ):
        # ---- SAFETY DEFAULTS (prevents crashes) ----
        financial_override = False
        override_reason = ""
        is_verified_financial = False
        financial_legitimacy_score = 0
        is_legit_financial_sender = False

        fraud_prob = float(probability[1])
        legit_prob = float(probability[0])

        if fraud_prob >= 0.9:
            risk = "🔴 CRITICAL"
            rec = "⛔ BLOCK - Extremely high confidence fraud"
        elif fraud_prob >= 0.7:
            risk = "🟠 HIGH"
            rec = "⚠️ WARN - High likelihood of fraud"
        elif fraud_prob >= 0.5:
            risk = "🟡 MEDIUM"
            rec = "⚡ ALERT - Suspicious patterns detected"
        elif fraud_prob >= 0.3:
            risk = "🟢 LOW-MEDIUM"
            rec = "👀 REVIEW - Some suspicious elements"
        else:
            risk = "🟢 LOW"
            rec = "✅ SAFE - Appears legitimate"

        indicators = []

        if is_verified_financial:
            indicators.append(
                f"✅ Verified financial institution ({financial_legitimacy_score}/8)"
            )
        elif is_legit_financial_sender:
            indicators.append(f"ℹ️ Known financial sender: {sender_id}")

        return {
            "is_fraud": bool(prediction),
            "fraud_probability": fraud_prob,
            "legitimate_probability": legit_prob,
            "risk_level": risk,
            "recommendation": rec,
            "message_type": message_type,
            "fraud_indicators": indicators,
            "model_used": f"{message_type}_model",
            "confidence": (
                "high"
                if abs(fraud_prob - 0.5) > 0.3
                else "medium"
                if abs(fraud_prob - 0.5) > 0.1
                else "low"
            ),
            "verified_financial": is_verified_financial,
            "financial_legitimacy_score": financial_legitimacy_score,
            "financial_override_applied": financial_override,
        }
