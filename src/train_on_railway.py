"""
FIXED: Railway Cloud Training Script
Trains BOTH transaction and promotion fraud models correctly
"""

import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

def train_transaction_model():
    """Train the transaction fraud detection model"""
    
    print("="*70)
    print("🏋️ TRAINING TRANSACTION FRAUD MODEL")
    print("="*70)
    
    # 1. Load transaction data
    data_path = "data/processed/mpesa_sms_features.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Transaction data not found: {data_path}")
        return False
    
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Loaded {len(df):,} transaction messages")
    
    # 2. Define features that ACTUALLY exist in transaction data
    TRANSACTION_FEATURES = [
        'message_length',
        'action_verb_count',
        'has_urgent',
        'exclamation_ratio',
        'transaction_completeness',
        'is_valid_sender',
        'has_link',
        'link_with_urgency',
        'link_without_transaction',
        'has_spelling_error',
        'soft_action_count',
        'authority_count'
    ]
    
    # Check which features actually exist
    available_features = [f for f in TRANSACTION_FEATURES if f in df.columns]
    missing_features = [f for f in TRANSACTION_FEATURES if f not in df.columns]
    
    print(f"\n✅ Available features: {len(available_features)}")
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        print("   (Using only available features)")
    
    # 3. Prepare data
    X = df[available_features + ['message_text']]
    y = df['is_fraud']
    
    print(f"\n📊 Training data:")
    print(f"   Total: {len(y):,}")
    print(f"   Fraud: {y.sum():,} ({y.mean()*100:.1f}%)")
    
    # 4. Build pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), available_features),
            ('text', TfidfVectorizer(
                max_features=300,
                ngram_range=(1, 2),
                stop_words='english'
            ), 'message_text')
        ]
    )
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=len(y[y==0]) / len(y[y==1]),
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    
    # 5. Train
    print("\n🔧 Training XGBoost model...")
    pipeline.fit(X, y)
    
    # 6. Save
    os.makedirs('models', exist_ok=True)
    model_path = 'models/fraud_detector_pipeline.joblib'
    joblib.dump(pipeline, model_path)
    print(f"✅ Transaction model saved: {model_path}")
    
    return True


def train_promotion_model():
    """Train the promotion fraud detection model"""
    
    print("\n" + "="*70)
    print("🏋️ TRAINING PROMOTION FRAUD MODEL")
    print("="*70)
    
    # 1. Load promotion data
    data_path = "data/processed/mpesa_promotion_features.csv"
    
    if not os.path.exists(data_path):
        print(f"⚠️  Promotion data not found: {data_path}")
        print("   Skipping promotion model (not critical)")
        return False
    
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Loaded {len(df):,} promotion messages")
    
    # 2. Define promotion-specific features
    PROMOTION_FEATURES = [
        'is_legit_sender', 'sender_is_number', 'sender_length',
        'suspicious_sender_name', 'has_link', 'link_count',
        'has_legit_domain', 'has_fraud_shortener',
        'has_ussd_code', 'has_mobile_number',
        'has_win_claim', 'win_claim_count',
        'has_prize_amount', 'max_prize_amount',
        'unrealistic_prize', 'has_guarantee',
        'has_payment_request', 'has_suspicious_fee',
        'urgency_count', 'has_urgency',
        'exclamation_count', 'message_length',
        'has_safaricom_brand', 'mentions_terms',
        'says_you_won', 'has_fraud_pattern',
        'legitimacy_score', 'fraud_risk_score'
    ]
    
    # Check which features exist
    available_features = [f for f in PROMOTION_FEATURES if f in df.columns]
    
    print(f"\n✅ Available features: {len(available_features)}")
    
    # 3. Prepare data
    X = df[available_features + ['message_text']]
    y = df['is_fraud']
    
    print(f"\n📊 Training data:")
    print(f"   Total: {len(y):,}")
    print(f"   Fraud: {y.sum():,} ({y.mean()*100:.1f}%)")
    
    # 4. Build pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), available_features),
            ('text', TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2
            ), 'message_text')
        ]
    )
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            scale_pos_weight=len(y[y==0]) / len(y[y==1]),
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    
    # 5. Train
    print("\n🔧 Training XGBoost model...")
    pipeline.fit(X, y)
    
    # 6. Save
    model_path = 'models/promotion_fraud_detector.joblib'
    joblib.dump(pipeline, model_path)
    print(f"✅ Promotion model saved: {model_path}")
    
    return True


def main():
    """Main training orchestrator"""
    
    print("\n" + "="*70)
    print("🚀 RAILWAY CLOUD TRAINING - M-PESA FRAUD DETECTION")
    print("="*70)
    
    # Check data directory
    if not os.path.exists("data/processed"):
        print("\n❌ ERROR: data/processed/ directory not found!")
        print("   Make sure you pushed the data folder to GitHub")
        return
    
    print("\n📂 Available data files:")
    for file in os.listdir("data/processed"):
        size = os.path.getsize(f"data/processed/{file}") / (1024*1024)
        print(f"   • {file} ({size:.1f} MB)")
    
    # Train both models
    transaction_success = train_transaction_model()
    promotion_success = train_promotion_model()
    
    # Summary
    print("\n" + "="*70)
    print("📋 TRAINING SUMMARY")
    print("="*70)
    
    if transaction_success:
        print("✅ Transaction Model: TRAINED")
    else:
        print("❌ Transaction Model: FAILED")
    
    if promotion_success:
        print("✅ Promotion Model: TRAINED")
    else:
        print("⚠️  Promotion Model: SKIPPED (optional)")
    
    if transaction_success:
        print("\n🎉 SUCCESS! At least one model trained successfully.")
        print("   Your bot will work with the transaction fraud detector.")
    else:
        print("\n❌ FAILURE! No models were trained.")
        print("   Check your data files and try again.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()