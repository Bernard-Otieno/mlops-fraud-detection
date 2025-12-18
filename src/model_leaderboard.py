import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline


# 1. Load data
df = pd.read_csv("data/processed/mpesa_sms_features.csv")
features = ['message_length', 'all_caps_count', 'exclamation_count', 'is_valid_sender',
            'sender_id_mismatch', 'has_link', 'is_phishy_link', 'urgent_without_transaction',
            'has_pin_word']  # Add your new ones like all_caps_ratio, etc.
X = df[features]
y = df['is_fraud']

# Train/test split (on original data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaler (only for numeric features)
scaler = StandardScaler()

# Models with SMOTE + scaling in pipeline
models = {
    "Logistic Regression": ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', scaler),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42))
    ]),
    "Random Forest": ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        ))
    ])
}

# Now training and evaluation
results = []
for name, pipeline in models.items():
    # Fit on training data (SMOTE only applied here)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

    # Cross-validation (now safe!)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    print(f"{name} - 5-fold CV F1 on train: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
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