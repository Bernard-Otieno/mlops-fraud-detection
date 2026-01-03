"""
M-PESA Promotion Fraud Detection - Model Training
Train and evaluate models for detecting fraudulent promotional messages
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix, roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature groups
NUMERIC_FEATURES = [
    # Sender features
    'is_legit_sender', 'sender_is_number', 'sender_length', 'suspicious_sender_name',
    
    # Link features
    'has_link', 'link_count', 'has_legit_domain', 'has_fraud_shortener',
    'has_typosquat_domain', 'link_at_end', 'link_at_start',
    
    # Contact features
    'has_ussd_code', 'ussd_code_count', 'has_sms_shortcode', 'sms_shortcode_count',
    'has_official_contact', 'has_mobile_number', 'mobile_number_count',
    
    # Prize features
    'has_win_claim', 'win_claim_count', 'has_prize_amount', 'max_prize_amount',
    'unrealistic_prize', 'extreme_prize', 'has_guarantee', 'guarantee_count',
    
    # Payment features
    'has_payment_request', 'suspicious_fee_count', 'has_suspicious_fee',
    'pay_to_claim_pattern',
    
    # Urgency features
    'urgency_count', 'has_urgency', 'high_urgency', 'has_time_pressure',
    'mentions_expiration',
    
    # Language features
    'exclamation_count', 'exclamation_density', 'excessive_exclamations',
    'question_count', 'caps_word_count', 'excessive_caps', 'emoji_count',
    'has_emoji', 'message_length', 'word_count',
    
    # Legitimacy features
    'safaricom_brand_count', 'has_safaricom_brand', 'mentions_terms',
    'says_stand_a_chance', 'says_you_won', 'mentions_optout', 'hashtag_count',
    'has_hashtag', 'says_dial', 'has_sms_instruction',
    
    # Scam patterns
    'has_click_bait', 'requests_verification', 'claims_authority',
    'fraud_pattern_count', 'has_fraud_pattern',
    
    # Composite features
    'win_fee_urgency_combo', 'link_payment_combo', 'unrealistic_prize_fee',
    'no_legit_contact_payment', 'fraud_link_win_combo', 'legitimacy_score',
    'high_legitimacy', 'fraud_risk_score', 'high_fraud_risk'
]

TEXT_FEATURE = 'message_text'

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

def load_data():
    """Load and prepare dataset"""
    
    print("="*70)
    print("📂 LOADING PROMOTION FRAUD DATASET")
    print("="*70)
    
    df = pd.read_csv('data/processed/mpesa_promotion_features.csv')
    
    print(f"\n✅ Loaded {len(df):,} promotional messages")
    print(f"   Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"   Legitimate: {(~df['is_fraud'].astype(bool)).sum():,} ({(~df['is_fraud'].astype(bool)).mean()*100:.1f}%)")
    
    # Check for missing features
    missing_features = [f for f in NUMERIC_FEATURES if f not in df.columns]
    if missing_features:
        print(f"\n⚠️  Warning: Missing features: {missing_features}")
        # Remove missing features
        NUMERIC_FEATURES[:] = [f for f in NUMERIC_FEATURES if f in df.columns]
    
    print(f"\n📊 Using {len(NUMERIC_FEATURES)} numeric features + text features")
    
    return df

# ============================================================================
# BUILD PREPROCESSING PIPELINE
# ============================================================================

def build_preprocessor():
    """Build feature preprocessing pipeline"""
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('text', TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2
            ), TEXT_FEATURE)
        ]
    )
    
    return preprocessor

# ============================================================================
# TRAIN AND EVALUATE MODELS
# ============================================================================

def train_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train multiple models and compare performance"""
    
    print("\n" + "="*70)
    print("🏋️  TRAINING MODELS")
    print("="*70)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=0.5,
            random_state=42
        ),
        
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=5,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        ),
        
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_leaf=3,
            min_samples_split=5,
            class_weight='balanced',
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
    }
    
    results = []
    trained_pipelines = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        
        # Build pipeline
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        })
        
        trained_pipelines[name] = pipeline
        
        print(f"   ✅ Accuracy: {accuracy:.3f}")
        print(f"   ✅ Precision: {precision:.3f}")
        print(f"   ✅ Recall: {recall:.3f}")
        print(f"   ✅ F1-Score: {f1:.3f}")
        print(f"   ✅ AUC: {auc:.3f}")
    
    return results, trained_pipelines

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

def display_leaderboard(results):
    """Display model comparison leaderboard"""
    
    print("\n" + "="*70)
    print("🏆 MODEL LEADERBOARD")
    print("="*70)
    
    leaderboard = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
    print("\n" + leaderboard.to_string(index=False))
    
    return leaderboard

def plot_results(results, save_path='promotion_model_comparison.png'):
    """Create visualization of model performance"""
    
    df_results = pd.DataFrame(results)
    
    # Melt for plotting
    df_melted = df_results.melt(
        id_vars='Model',
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        var_name='Metric',
        value_name='Score'
    )
    
    # Create plot
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Bar plot
    ax = sns.barplot(
        data=df_melted,
        x='Metric',
        y='Score',
        hue='Model',
        palette='viridis'
    )
    
    plt.title('Promotion Fraud Detection - Model Comparison', fontsize=16, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {save_path}")
    plt.close()

def analyze_best_model(best_model_name, pipeline, X_test, y_test):
    """Detailed analysis of best performing model"""
    
    print("\n" + "="*70)
    print(f"🔍 DETAILED ANALYSIS: {best_model_name}")
    print("="*70)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Legitimate', 'Fraud'],
        zero_division=0
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted Legit', 'Predicted Fraud'],
        yticklabels=['Actual Legit', 'Actual Fraud']
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{best_model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('promotion_confusion_matrix.png', dpi=300)
    print("\n📊 Confusion matrix saved to: promotion_confusion_matrix.png")
    plt.close()
    
    # Feature importance (if tree-based)
    if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
        print("\n📊 Top 15 Most Important Features:")
        
        # Get feature names
        num_features = NUMERIC_FEATURES
        text_features = pipeline.named_steps['prep'].named_transformers_['text'].get_feature_names_out()
        all_feature_names = num_features + list(text_features)
        
        # Get importances
        importances = pipeline.named_steps['clf'].feature_importances_
        
        # Create dataframe
        feature_importance = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)
        
        print(feature_importance.to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=feature_importance,
            x='importance',
            y='feature',
            palette='viridis'
        )
        plt.title(f'{best_model_name} - Top 15 Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('promotion_feature_importance.png', dpi=300)
        print("\n📊 Feature importance plot saved to: promotion_feature_importance.png")
        plt.close()

# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(pipeline, model_name, leaderboard, feature_list):
    """Save trained model and metadata"""
    
    print("\n" + "="*70)
    print("💾 SAVING MODEL")
    print("="*70)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save pipeline
    model_path = 'models/promotion_fraud_detector.joblib'
    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save feature list
    features_path = 'models/promotion_feature_list.txt'
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_list))
    print(f"✅ Feature list saved to: {features_path}")
    
    # Save metadata
    best_performance = leaderboard.iloc[0]
    metadata = {
        'model_type': model_name,
        'features': feature_list,
        'feature_count': len(feature_list),
        'training_date': datetime.now().isoformat(),
        'performance': {
            'accuracy': float(best_performance['Accuracy']),
            'precision': float(best_performance['Precision']),
            'recall': float(best_performance['Recall']),
            'f1_score': float(best_performance['F1-Score']),
            'auc': float(best_performance['AUC'])
        },
        'model_purpose': 'promotion_fraud_detection'
    }
    
    metadata_path = 'models/promotion_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_path}")
    
    print("\n" + "="*70)
    print("✨ MODEL TRAINING COMPLETE!")
    print("="*70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("🚀 M-PESA PROMOTION FRAUD DETECTION - MODEL TRAINING")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Prepare features and target
    X = df[NUMERIC_FEATURES + [TEXT_FEATURE]]
    y = df['is_fraud']
    
    # Train/test split
    print("\n" + "="*70)
    print("✂️  SPLITTING DATA")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\n✅ Training set: {len(X_train):,} samples")
    print(f"   Fraud: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    print(f"✅ Test set: {len(X_test):,} samples")
    print(f"   Fraud: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
    
    # Build preprocessor
    preprocessor = build_preprocessor()
    
    # Train models
    results, trained_pipelines = train_models(
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # Display leaderboard
    leaderboard = display_leaderboard(results)
    
    # Plot results
    plot_results(results)
    
    # Get best model
    best_model_name = leaderboard.iloc[0]['Model']
    best_pipeline = trained_pipelines[best_model_name]
    
    # Analyze best model
    analyze_best_model(best_model_name, best_pipeline, X_test, y_test)
    
    # Save model
    save_model(best_pipeline, best_model_name, leaderboard, NUMERIC_FEATURES + [TEXT_FEATURE])
    
    # Final summary
    print("\n" + "="*70)
    print("📋 FINAL SUMMARY")
    print("="*70)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📊 F1-Score: {leaderboard.iloc[0]['F1-Score']:.3f}")
    print(f"🎯 Accuracy: {leaderboard.iloc[0]['Accuracy']:.3f}")
    print(f"⚡ Precision: {leaderboard.iloc[0]['Precision']:.3f}")
    print(f"🔍 Recall: {leaderboard.iloc[0]['Recall']:.3f}")
    print(f"📈 AUC: {leaderboard.iloc[0]['AUC']:.3f}")
    
    print("\n" + "="*70)
    print("✨ ALL DONE! Model ready for deployment.")
    print("="*70)

if __name__ == "__main__":
    main()