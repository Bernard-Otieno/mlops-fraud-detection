import pandas as pd

df = pd.read_csv('data/processed/my_features.csv')

pin_messages = df[df['has_pin_word'] == 1]
print(f"Fraud rate in PIN messages: {pin_messages['is_fraud'].mean()*100:.1f}%")
print(f"Overall fraud rate: {df['is_fraud'].mean()*100:.1f}%")
print(f"Risk multiplier: {pin_messages['is_fraud'].mean() / df['is_fraud'].mean():.1f}x")