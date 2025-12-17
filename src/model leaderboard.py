import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. SETUP DATA (Assuming your features are ready)
df = pd.read_csv("data/processed/mpesa_sms_features.csv")
features = ['message_length', 'has_link', 'has_pin_word', 'urgent_without_transaction', 'is_valid_sender']
X = df[features]
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. RUN TOURNAMENT
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
os.makedirs("models", exist_ok=True)
for name, model in models.items():
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    threshold = 0.35
    y_pred = (y_probs >= threshold).astype(int)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print("\n--- FINAL LEADERBOARD ---")
print(results_df)

# 7. VISUALIZATION (Comparing the Models)
sns.set_theme(style="whitegrid")
results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(12, 7))
plot = sns.barplot(data=results_melted, x="Metric", y="Score", hue="Model", palette="magma")
plt.title("MLOps Performance Tournament: Model Comparison", fontsize=16)
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add score labels on top of bars
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=8)

plt.tight_layout()
plt.savefig("model_comparison_plot.png")
print("\n📊 Success! Plot saved as 'model_comparison_plot.png'")
# 4. PLOT 2: FEATURE IMPORTANCE (From the Random Forest)
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='magma')
plt.title("Winning Model: Feature Importance Rankings")
plt.savefig("feature_importance.png", bbox_inches='tight')

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Legit', 'Predicted Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Where is the model failing?')
plt.savefig("confusion_matrix.png")