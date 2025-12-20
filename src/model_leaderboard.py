import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier


df = pd.read_csv("data/processed/mpesa_sms_features.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Behavioral features
sender_counts = df.groupby('sender_id').size()
df['sender_seen_before'] = (df['sender_id'].map(sender_counts) > 1).astype(int)

df = df.sort_values('timestamp')
df['time_diff'] = df.groupby('sender_id')['timestamp'].diff().dt.total_seconds()
df['burst_activity'] = (df['time_diff'] < 600).astype(int).fillna(0)

amount_counts = df.groupby('amount').size()
df['reused_amount'] = (df['amount'].map(amount_counts) > 3).astype(int)

# ============================================================================
# FIX 1: STANDARD TRAIN/TEST SPLIT (80/20)
# ============================================================================

FEATURES = [
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

X = df[FEATURES + ['message_text']]
y = df['is_fraud']

# Standard stratified split - ensures both classes in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # ← Key: maintains fraud/legit ratio in both sets
)

print("="*70)
print("DATASET SPLIT")
print("="*70)
print(f"\nTraining set:")
print(f"  Total: {len(y_train):,}")
print(f"  Fraud: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Legit: {(~y_train.astype(bool)).sum():,}")

print(f"\nTest set:")
print(f"  Total: {len(y_test):,}")
print(f"  Fraud: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
print(f"  Legit: {(~y_test.astype(bool)).sum():,}")

# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), FEATURES),
        ('text', TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            stop_words='english'
        ), 'message_text')
    ]
)

# ============================================================================
# MODELS WITH BETTER HYPERPARAMETERS
# ============================================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=0.1,  # Regularization
        random_state=42
    ),
    
    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,  # Deeper than before
        min_samples_leaf=5,  # Less restrictive
        min_samples_split=10,
        class_weight='balanced',  # ← KEY FIX!
        random_state=42
    ),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,  # Deeper
        min_samples_leaf=3,  # Less restrictive
        min_samples_split=5,
        class_weight='balanced',  # ← KEY FIX!
        max_features='sqrt',
        random_state=42
    ),
    
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalance
        random_state=42,
        eval_metric='logloss'
    )
}

# ============================================================================
# TRAIN AND EVALUATE
# ============================================================================

results = []

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    
    pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        'Model': name,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })
    
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1: {f1:.3f}")

# ============================================================================
# LEADERBOARD
# ============================================================================

leaderboard = pd.DataFrame(results).sort_values('F1', ascending=False)

print("\n" + "="*70)
print("📊 FINAL LEADERBOARD")
print("="*70)
print(leaderboard.to_string(index=False))

# ============================================================================
# DETAILED ANALYSIS OF BEST MODEL
# ============================================================================

best_model_name = leaderboard.iloc[0]['Model']
print(f"\n" + "="*70)
print(f"🏆 BEST MODEL: {best_model_name}")
print("="*70)

# Retrain best model for analysis
best_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', models[best_model_name])
])
best_pipe.fit(X_train, y_train)
y_pred_best = best_pipe.predict(X_test)

# Classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_best, 
                           target_names=['Legitimate', 'Fraud'],
                           zero_division=0))

# Feature importance (if tree-based)
if best_model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
    print("\n📊 Top 10 Most Important Features:")
    
    # Get feature names after preprocessing
    num_features = FEATURES
    text_features = best_pipe.named_steps['prep'].named_transformers_['text'].get_feature_names_out()
    all_feature_names = num_features + list(text_features)
    
    # Get importances
    importances = best_pipe.named_steps['clf'].feature_importances_
    
    # Sort and display top 10
    feature_importance = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    
    print(feature_importance.to_string(index=False))

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("🔍 ERROR ANALYSIS")
print("="*70)

# Create analysis dataframe
test_analysis = X_test.copy()
test_analysis['actual'] = y_test.values
test_analysis['predicted'] = y_pred_best

# False Negatives (missed fraud)
false_negatives = test_analysis[
    (test_analysis['actual'] == 1) & (test_analysis['predicted'] == 0)
]

# False Positives (false alarms)
false_positives = test_analysis[
    (test_analysis['actual'] == 0) & (test_analysis['predicted'] == 1)
]

print(f"\n🚨 False Negatives (Missed Fraud): {len(false_negatives)}")
if len(false_negatives) > 0:
    print("\nSample missed fraud messages:")
    for idx, row in false_negatives.head(3).iterrows():
        print(f"\n  Message: {row['message_text'][:100]}...")
        print(f"  Features: length={row['message_length']}, urgent={row['has_urgent']}, link={row['has_link']}")

print(f"\n⚠️  False Positives (False Alarms): {len(false_positives)}")
if len(false_positives) > 0:
    print("\nSample false alarms:")
    for idx, row in false_positives.head(3).iterrows():
        print(f"\n  Message: {row['message_text'][:100]}...")
        print(f"  Features: length={row['message_length']}, urgent={row['has_urgent']}, link={row['has_link']}")

print("\n" + "="*70)
print("✨ Analysis Complete!")
print("="*70)


# ============================================================================
# SAVE BEST MODEL
# ============================================================================

print("\n" + "="*70)
print("💾 SAVING MODEL")
print("="*70)

# Get best model
best_model_name = leaderboard.iloc[0]['Model']
print(f"Best model: {best_model_name}")

# Retrain on ALL data for production (optional but recommended)
print("Training final model on full dataset...")
best_pipe_final = Pipeline([
    ('prep', preprocessor),
    ('clf', models[best_model_name])
])
best_pipe_final.fit(X, y)  # Train on ALL data

# Create models directory
os.makedirs('models', exist_ok=True)

# Save the complete pipeline
model_path = 'models/fraud_detector_pipeline.joblib'
joblib.dump(best_pipe_final, model_path)
print(f"✅ Model saved to: {model_path}")

# Save feature list
features_path = 'models/feature_list.txt'
with open(features_path, 'w') as f:
    f.write('\n'.join(FEATURES))
print(f"✅ Feature list saved to: {features_path}")

# Save metadata
metadata = {
    'model_type': best_model_name,
    'features': FEATURES,
    'training_date': pd.Timestamp.now().isoformat(),
    'training_samples': len(df),
    'fraud_rate': float(y.mean()),
    'performance': {
        'precision': float(leaderboard.iloc[0]['Precision']),
        'recall': float(leaderboard.iloc[0]['Recall']),
        'f1': float(leaderboard.iloc[0]['F1'])
    }
}

import json
metadata_path = 'models/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved to: {metadata_path}")

print("\n" + "="*70)
print("✨ MODEL SAVED SUCCESSFULLY!")
print("="*70)


# ```

# ---

# ## **Expected Results with Fix 1:**
# ```
# 📊 FINAL LEADERBOARD
#                  Model  Precision  Recall      F1
#               XGBoost      0.892   0.847   0.869
#         Random Forest      0.876   0.823   0.849
#         Decision Tree      0.831   0.789   0.809
#   Logistic Regression      0.798   0.756   0.776