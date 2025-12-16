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
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Fraud distribution
df['is_fraud'].value_counts().plot(
    kind='bar',
    ax=axs[0],
    color=['green', 'red']
)
axs[0].set_title('Fraud vs Legitimate')
axs[0].set_xlabel('Is Fraud')
axs[0].set_ylabel('Count')

# Plot 2: Valid sender vs Fraud
fraud_by_sender = df.groupby(['is_valid_sender', 'is_fraud']).size().unstack()
fraud_by_sender.plot(
    kind='bar',
    ax=axs[1],
    color=['green', 'red']
)
axs[1].set_title('Valid Sender vs Fraud')
axs[1].set_xlabel('Is Valid Sender')
axs[1].legend(['Legitimate', 'Fraud'])

# Plot 3: Has Link vs Fraud
has_link = df.groupby(['has_link', 'is_fraud']).size().unstack()
has_link.plot(
    kind='bar',
    ax=axs[2],
    color=['green', 'red']
)
axs[2].set_title('Has Link vs Fraud')
axs[2].set_xlabel('Has Link')
axs[2].legend(['Legitimate', 'Fraud'])

plt.tight_layout()
plt.savefig('my_first_eda.png')
plt.show()

# Question 4: Distribution of message lengths vs fraud
fig, ax = plt.subplots(figsize=(6, 4))

df['message_length'].plot(kind='hist', bins=30, ax=ax)
df[df['is_fraud'] == 1]['message_length'].plot(
    kind='hist', bins=30, ax=ax, color='red', alpha=0.5
)
ax.set_title('Distribution of Message Lengths vs Fraud')

plt.tight_layout()
plt.show()



# print("\n✅ Saved plot to my_first_eda.png")