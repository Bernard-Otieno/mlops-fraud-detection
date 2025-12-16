import pandas as pd
import matplotlib.pyplot as plt

# Load your featured data
df = pd.read_csv('data/processed/my_features.csv')

# Question 1: How many fraud vs legitimate?
print("Fraud distribution:")
print(df['is_fraud'].value_counts())
print(f"Fraud rate: {df['is_fraud'].mean() * 100:.1f}%")

# Question 2: Are fraud messages longer or shorter?
print("\nMessage length by fraud status:")
print(df.groupby('is_fraud')['message_length'].mean())

# Question 3: Visualize it!
plt.figure(figsize=(10, 5))

# Plot 1: Fraud distribution
plt.subplot(1, 2, 1)
df['is_fraud'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Fraud vs Legitimate')
plt.xlabel('Is Fraud')
plt.ylabel('Count')

# Plot 2: Valid sender by fraud
plt.subplot(1, 2, 2)
fraud_by_sender = df.groupby(['is_valid_sender', 'is_fraud']).size().unstack()
fraud_by_sender.plot(kind='bar', color=['green', 'red'])
plt.title('Valid Sender vs Fraud')
plt.xlabel('Is Valid Sender')
plt.legend(['Legitimate', 'Fraud'])

plt.tight_layout()
plt.savefig('my_first_eda.png')
plt.show()

print("\n✅ Saved plot to my_first_eda.png")