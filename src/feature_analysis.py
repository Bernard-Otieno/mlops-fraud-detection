import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/my_features.csv')

# Analyze the 3 new features
new_features = ['has_link', 'exclamation_count', 'has_urgent_word']

print("="*70)
print("📊 NEW FEATURES ANALYSIS")
print("="*70)

for feature in new_features:
    print(f"\n🔍 {feature}")
    print("-"*70)
    
    if feature == 'exclamation_count':
        # Continuous feature
        print(f"Fraud messages - Average: {df[df['is_fraud']==1][feature].mean():.2f}")
        print(f"Legit messages - Average: {df[df['is_fraud']==0][feature].mean():.2f}")
    else:
        # Binary feature
        has_feature = df[df[feature] == 1]
        if len(has_feature) > 0:
            print(f"Messages with feature: {len(has_feature):,}")
            print(f"Fraud rate: {has_feature['is_fraud'].mean()*100:.1f}%")
            print(f"Risk: {has_feature['is_fraud'].mean() / df['is_fraud'].mean():.1f}x baseline")
        else:
            print("No messages found with this feature!")

# Correlation summary
print("\n" + "="*70)
print("📈 ALL FEATURES - CORRELATION SUMMARY")
print("="*70)

all_features = ['is_valid_sender', 'message_length', 'has_pin_word', 
                'amount_count', 'has_confirmed', 'has_link', 
                'exclamation_count', 'has_urgent_word']

correlations = []
for feature in all_features:
    corr = df[[feature, 'is_fraud']].corr().iloc[0, 1]
    correlations.append((feature, corr))

# Sort by absolute value
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nRanked by strength (absolute value):")
for i, (feature, corr) in enumerate(correlations, 1):
    strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
    print(f"  {i}. {feature:25s}: {corr:+.3f}  [{strength}]")