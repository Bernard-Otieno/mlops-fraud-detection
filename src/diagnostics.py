# diagnostics.py
# Centralized diagnostics for M-PESA SMS fraud detection dataset
# Run this after generate_mpesa_sms.py and feature_extractor.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output folder for plots
os.makedirs("diagnostics_plots", exist_ok=True)

# 1. Load the final processed data
df = pd.read_csv("data/processed/mpesa_sms_features.csv")

print(f"Loaded {len(df):,} messages ({df['is_fraud'].mean()*100:.1f}% fraud)")

# ==================== BASIC STATS ====================
print("\n=== Message Length Stats ===")
print(df['message_length'].describe())
print("\nBy Class:")
print(df.groupby('is_fraud')['message_length'].describe())

# Quantiles for deeper insight
print("\nLegit Message Length Quantiles (0%, 25%, 50%, 75%, 100%):")
print(np.percentile(df[df['is_fraud']==0]['message_length'], [0, 25, 50, 75, 100]))
print("Fraud Message Length Quantiles:")
print(np.percentile(df[df['is_fraud']==1]['message_length'], [0, 25, 50, 75, 100]))

# ==================== VISUALIZATIONS ====================
# Plot 1: Message Length Distribution (your original)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='message_length', hue='is_fraud', kde=True, bins=30)
plt.title('Message Length Distribution by Fraud Class')
plt.xlabel('Message Length (characters)')
plt.savefig('diagnostics_plots/message_length_dist.png')
plt.close()
print("\nSaved: message_length_dist.png")

# Plot 2: Boxplot for clearer separation
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='is_fraud', y='message_length')
plt.title('Message Length Boxplot by Class')
plt.xticks([0, 1], ['Legitimate', 'Fraudulent'])
plt.savefig('diagnostics_plots/message_length_boxplot.png')
plt.close()
print("Saved: message_length_boxplot.png")

# Plot 3: Top features correlation heatmap
features = ['message_length', 'all_caps_count', 'exclamation_count', 'is_valid_sender',
            'sender_id_mismatch', 'has_link', 'is_phishy_link', 'urgent_without_transaction',
            'has_pin_word']
corr_matrix = df[features + ['is_fraud']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlations with is_fraud')
plt.savefig('diagnostics_plots/feature_correlations.png')
plt.close()
print("Saved: feature_correlations.png")

# ==================== DEEPER DIAGNOSES ====================
# Diagnosis 1: How separable are classes by length alone?
from sklearn.metrics import roc_auc_score
auc_length = roc_auc_score(df['is_fraud'], df['message_length'])
print(f"\nAUC if using ONLY message_length as predictor: {auc_length:.3f}")
print("(>0.7 is strong separation — explains high feature importance!)")

# Diagnosis 2: "Impossible" rows — Legit messages with PIN requests (data leak?)
impossible_pin = df[(df['is_fraud'] == 0) & (df['has_pin_word'] == 1)]
print(f"\nLegit messages mentioning 'PIN' (potential leak): {len(impossible_pin)}")
if len(impossible_pin) > 0:
    print("Samples:")
    print(impossible_pin[['message_text', 'message_length']].head(5))

# Diagnosis 3: Links in legit vs fraud (since we added realistic promo links)
print("\nLink Analysis:")
print(f"Legitimate with links: {df[(df['is_fraud']==0) & (df['has_link']==1)].shape[0]}")
print(f"Fraudulent with links: {df[(df['is_fraud']==1) & (df['has_link']==1)].shape[0]}")
print(f"Legitimate with phishy links: {df[(df['is_fraud']==0) & (df['is_phishy_link']==1)].shape[0]} (should be 0!)")

# Diagnosis 4: Sample extreme messages
print("\nSample SHORT fraudulent messages (<100 chars):")
short_fraud = df[(df['is_fraud']==1) & (df['message_length'] < 100)].sample(3, random_state=42)
for _, row in short_fraud.iterrows():
    print(f"Length {row['message_length']}: {row['message_text']}")

print("\nSample LONG legitimate messages (>200 chars):")
long_legit = df[(df['is_fraud']==0) & (df['message_length'] > 200)].sample(3, random_state=42)
for _, row in long_legit.iterrows():
    print(f"Length {row['message_length']}: {row['message_text']}")

# Optional: Save a full diagnostics report
with open("diagnostics_plots/report.txt", "w") as f:
    f.write("M-PESA SMS Fraud Diagnostics Report\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Fraud rate: {df['is_fraud'].mean()*100:.1f}%\n")
    f.write(f"Length AUC: {auc_length:.3f}\n")
print("\nAll diagnostics complete! Check diagnostics_plots/ folder.")

# In diagnostics.py, replace the AUC line with:
from sklearn.metrics import roc_auc_score

# Compute properly directional AUC
length_scores = df['message_length']
if df['message_length'].corr(df['is_fraud']) < 0:
    length_scores = -length_scores  # Flip if negative correlation

auc_length = roc_auc_score(df['is_fraud'], length_scores)
print(f"\nDirectional Length AUC (higher = better separation): {auc_length:.3f}")