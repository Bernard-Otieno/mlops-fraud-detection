"""
Complete Model Fix - Both Transaction AND Promotion Models
This will fix both models so your WhatsApp bot works with ALL message types
"""

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import json

print("="*70)
print("🔧 FIXING BOTH MODELS - TRANSACTION + PROMOTION")
print("="*70)

# ============================================================================
# FIX 1: TRANSACTION MODEL
# ============================================================================

print("\n" + "="*70)
print("📊 PART 1: FIXING TRANSACTION MODEL")
print("="*70)

# Load transaction data
print("\n📂 Loading transaction data...")
df_trans = pd.read_csv("data/processed/mpesa_sms_features.csv")
print(f"✅ Loaded {len(df_trans):,} transaction messages")

# Prepare features
df_trans['timestamp'] = pd.to_datetime(df_trans['timestamp'])

sender_counts = df_trans.groupby('sender_id').size()
df_trans['sender_seen_before'] = (df_trans['sender_id'].map(sender_counts) > 1).astype(int)

df_trans = df_trans.sort_values('timestamp')
df_trans['time_diff'] = df_trans.groupby('sender_id')['timestamp'].diff().dt.total_seconds()
df_trans['burst_activity'] = (df_trans['time_diff'] < 600).astype(int).fillna(0)

amount_counts = df_trans.groupby('amount').size()
df_trans['reused_amount'] = (df_trans['amount'].map(amount_counts) > 3).astype(int)

TRANS_FEATURES = [
    'message_length', 'action_verb_count', 'has_urgent',
    'exclamation_ratio', 'reused_amount', 'transaction_completeness',
    'is_valid_sender', 'has_link', 'link_with_urgency',
    'link_without_transaction', 'has_spelling_error',
    'soft_action_count', 'authority_count', 'sender_seen_before',
    'burst_activity'
]

X_trans = df_trans[TRANS_FEATURES + ['message_text']]
y_trans = df_trans['is_fraud']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_trans, y_trans, test_size=0.2, random_state=42, stratify=y_trans
)

print(f"\n✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Build pipeline with FIX
print("\n🏗️ Building transaction pipeline...")
trans_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), TRANS_FEATURES),
        ('text', TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            stop_words='english'
        ), 'message_text')
    ],
    verbose_feature_names_out=False  # 🔑 KEY FIX
)

trans_model = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=3,
    min_samples_split=5, class_weight='balanced',
    max_features='sqrt', random_state=42
)

trans_pipeline = Pipeline([
    ('prep', trans_preprocessor),
    ('clf', trans_model)
])

# Train
print("🏋️ Training transaction model...")
trans_pipeline.fit(X_train, y_train)

# Test
from sklearn.metrics import precision_score, recall_score, f1_score
y_pred = trans_pipeline.predict(X_test)
trans_f1 = f1_score(y_test, y_pred, zero_division=0)
print(f"✅ Transaction F1-Score: {trans_f1:.3f}")

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(trans_pipeline, 'models/fraud_detector_pipeline.joblib')
print("✅ Transaction model saved!")

trans_metadata = {
    'model_type': 'Random Forest',
    'features': TRANS_FEATURES,
    'training_date': datetime.now().isoformat(),
    'performance': {'f1': float(trans_f1)},
    'model_purpose': 'transaction_fraud_detection'
}
with open('models/model_metadata.json', 'w') as f:
    json.dump(trans_metadata, f, indent=2)

# ============================================================================
# FIX 2: PROMOTION MODEL
# ============================================================================

print("\n" + "="*70)
print("🎁 PART 2: FIXING PROMOTION MODEL")
print("="*70)

# Load promotion data
print("\n📂 Loading promotion data...")
try:
    df_promo = pd.read_csv("data/processed/mpesa_promotion_features.csv")
    print(f"✅ Loaded {len(df_promo):,} promotion messages")
except FileNotFoundError:
    print("⚠️  Promotion data not found. Generating it...")
    import subprocess
    subprocess.run(["python", "src/generate_promotions.py"])
    subprocess.run(["python", "src/promotion_feature_extractor.py"])
    df_promo = pd.read_csv("data/processed/mpesa_promotion_features.csv")
    print(f"✅ Generated and loaded {len(df_promo):,} promotion messages")

# Promotion features (all numeric features from your extractor)
PROMO_FEATURES = [
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

# Check which features exist
available_features = [f for f in PROMO_FEATURES if f in df_promo.columns]
print(f"\n✅ Using {len(available_features)}/{len(PROMO_FEATURES)} promotion features")

X_promo = df_promo[available_features + ['message_text']]
y_promo = df_promo['is_fraud']

# Split
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_promo, y_promo, test_size=0.2, random_state=42, stratify=y_promo
)

print(f"✅ Train: {len(X_train_p):,} | Test: {len(X_test_p):,}")

# Build pipeline with FIX
print("\n🏗️ Building promotion pipeline...")
promo_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), available_features),
        ('text', TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        ), 'message_text')
    ],
    verbose_feature_names_out=False  # 🔑 KEY FIX
)

promo_model = RandomForestClassifier(
    n_estimators=200, max_depth=25, min_samples_leaf=3,
    min_samples_split=5, class_weight='balanced',
    max_features='sqrt', random_state=42
)

promo_pipeline = Pipeline([
    ('prep', promo_preprocessor),
    ('clf', promo_model)
])

# Train
print("🏋️ Training promotion model...")
promo_pipeline.fit(X_train_p, y_train_p)

# Test
y_pred_p = promo_pipeline.predict(X_test_p)
promo_f1 = f1_score(y_test_p, y_pred_p, zero_division=0)
print(f"✅ Promotion F1-Score: {promo_f1:.3f}")

# Save
joblib.dump(promo_pipeline, 'models/promotion_fraud_detector.joblib')
print("✅ Promotion model saved!")

promo_metadata = {
    'model_type': 'Random Forest',
    'features': available_features,
    'feature_count': len(available_features),
    'training_date': datetime.now().isoformat(),
    'performance': {'f1_score': float(promo_f1)},
    'model_purpose': 'promotion_fraud_detection'
}
with open('models/promotion_model_metadata.json', 'w') as f:
    json.dump(promo_metadata, f, indent=2)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✨ BOTH MODELS FIXED SUCCESSFULLY!")
print("="*70)

print(f"""
📊 RESULTS:
   Transaction Model: F1 = {trans_f1:.3f} ✅
   Promotion Model:   F1 = {promo_f1:.3f} ✅

📁 FILES SAVED:
   ✅ models/fraud_detector_pipeline.joblib (Transaction)
   ✅ models/model_metadata.json
   ✅ models/promotion_fraud_detector.joblib (Promotion)
   ✅ models/promotion_model_metadata.json

🚀 READY TO USE:
   Your WhatsApp bot will now handle BOTH message types!
   
   Test it:
   1. Transaction fraud: "URGENT verify your account"
   2. Promotion fraud:   "You won Ksh500,000! Click here"
""")

print("="*70)
print("💡 Now restart your WhatsApp bot!")
print("="*70)