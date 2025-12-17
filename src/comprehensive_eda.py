import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("data/processed/mpesa_sms_features.csv")

# shape = df.shape
# columns = df.columns
# types = df.dtypes
# fraud_percentage = df["is_fraud"].value_counts(normalize=True) * 100
# message_length_stats = df["message_length"].describe()
# message_length_by_fraud = df.groupby("is_fraud")["message_length"].mean()
# exclamation_count_by_fraud = df.groupby("is_fraud")["exclamation_count"].mean()
# fraudulent_messages = df[df["is_fraud"] == 1]["message_text"].sample(5).tolist()
# legitimate_messages = df[df["is_fraud"] == 0]["message_text"].sample(5).tolist()
# has_link_fraud_rate = df.groupby("has_link")["is_fraud"].mean()
# link_with_urgency_fraud_rate = df.groupby("link_with_urgency")["is_fraud"].mean()
# has_urgent_word_fraud_rate = df.groupby("has_urgent_word")["is_fraud"].mean()
# has_pin_word_fraud_rate = df.groupby("has_pin_word")["is_fraud"].mean()
# urgent_without_transaction_fraud_rate = df.groupby("urgent_without_transaction")["is_fraud"].mean()
# pin_urgent_crosstab = pd.crosstab(
#     df["has_pin_word"],
#     df["urgent_without_transaction"],
#     values=df["is_fraud"],
#     aggfunc="mean"
# )


# print("Dataset Shape:", shape)
# print("\nColumns and Data Types:", columns.to_list(), types.to_list())
# print("\nFraud Percentage:\n", fraud_percentage)
# print("\nMessage Length Statistics:\n", message_length_stats)   
# print("\nAverage Message Length by Fraud Status:\n", message_length_by_fraud)
# print("\nAverage Exclamation Count by Fraud Status:\n", exclamation_count_by_fraud)
# print("\nSample Fraudulent Messages:")
# for msg in fraudulent_messages:
#     print("-", msg)
# print("\nSample Legitimate Messages:")
# for msg in legitimate_messages:
#     print("-", msg) 
# print("\nFraud Rate by Has Link:\n", has_link_fraud_rate)
# print("\nFraud Rate by Link with Urgency:\n", link_with_urgency_fraud_rate)
# print("\nFraud Rate by Has Urgent Word:\n", has_urgent_word_fraud_rate)
# print("\nFraud Rate by Has PIN Word:\n", has_pin_word_fraud_rate)  
# print("\nFraud Rate by Urgent without Transaction:\n", urgent_without_transaction_fraud_rate)
# print("\nCrosstab of Has PIN Word vs Urgent without Transaction:\n", pin_urgent_crosstab)


sns.set(style="whitegrid")

# Plot 1: Target Balance
plt.figure(figsize=(6,4))
sns.countplot(x="is_fraud", data=df, palette='Set2')
plt.title("Fraud vs Legitimate Messages")
plt.show()



# Feature Importance (correlation)
plt.figure(figsize=(10,8))
cols = ['is_fraud', 'message_length', 'has_link', 'has_shortened_link', 'has_pin_word', 'urgent_without_transaction']
corr = df[cols].corr()
sns.heatmap(corr, annot=True, cmap='RdBu', center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig('figures/feature_correlation_heatmap.png')
plt.show()

# --- GRAPH 3: DATA DISTRIBUTION ---
# Do fraud messages have a different 'shape' than real ones?
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='message_length', hue='is_fraud', fill=True)
plt.title("MLOps Check: Distribution of Message Length")
plt.show()