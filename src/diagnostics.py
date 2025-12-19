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

df['fraud_type'].value_counts()
print(df['fraud_type'].value_counts())


df.groupby('fraud_type')['is_fraud'].mean().sort_values()
print(df.groupby('fraud_type')['is_fraud'].mean().sort_values())


df.groupby('fraud_type')['message_length'].mean().sort_values()
print(df.groupby('fraud_type')['message_length'].mean().sort_values())



df.groupby('fraud_type')['has_link'].mean().sort_values(ascending=False)
print(df.groupby('fraud_type')['has_link'].mean().sort_values(ascending=False)) 



NEW_FEATURES = [
    'urgent_density',
    'action_verb_count',
    'transaction_completeness',
    'all_caps_ratio',
    'exclamation_ratio'
]
df.groupby('fraud_type')[NEW_FEATURES].mean()
print(df.groupby('fraud_type')[NEW_FEATURES].mean())

FEATURES = NEW_FEATURES + [
    'message_length',   
    'link_count',
    'has_shortener',
    'link_with_urgency',
    'sender_suspicious',
    'is_phishy_link',
    'has_spelling_error',
]
df.groupby('is_fraud')[FEATURES].mean()
print(df.groupby('is_fraud')[FEATURES].mean())