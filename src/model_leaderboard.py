import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

FEATURES = [
    'urgent_density',
    'action_verb_count',
    'transaction_completeness',
    'all_caps_ratio',
    'exclamation_ratio',
    'authority_density',
]
HOLDOUT_FRAUD = 'smart_social'
df = pd.read_csv("data/processed/mpesa_sms_features.csv")

train_df = df[
    (df['is_fraud'] == 0) |
    ((df['is_fraud'] == 1) & (df['fraud_type'] != HOLDOUT_FRAUD))
]

test_df = df[
    (df['is_fraud'] == 0) |
    (df['fraud_type'] == HOLDOUT_FRAUD)
]

X_train = train_df[FEATURES + ['message_text']]
y_train = train_df['is_fraud']

X_test  = test_df[FEATURES + ['message_text']]
y_test  = test_df['is_fraud']

print("Test fraud count:")
print(y_test.value_counts())

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), FEATURES),
        ('text', TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            stop_words='english'
        ), 'message_text')
    ]
)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=10,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
}
results = []


for name, model in models.items():
    pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    results.append({
        'Model': name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred)
    })
print("Predictions distribution:")
print(pd.Series(y_pred).value_counts())
precision_score(y_test, y_pred, zero_division=0)
recall_score(y_test, y_pred, zero_division=0)
f1_score(y_test, y_pred, zero_division=0)

print(precision_score(y_test, y_pred, zero_division=0))
print(recall_score(y_test, y_pred, zero_division=0))
print(f1_score(y_test, y_pred, zero_division=0))

# leaderboard = pd.DataFrame(results).sort_values('F1', ascending=False)
# print("\n📊 MODEL LEADERBOARD (Generalization Test)")
# print(leaderboard)


# best_pipe = pipe  # Logistic Regression
# test_df = df[
#     (df['is_fraud'] == 0) |
#     (df['fraud_type'] == HOLDOUT_FRAUD)
# ].copy()

# test_df['pred'] = best_pipe.predict(X_test)

# false_positives = test_df[
#     (test_df['pred'] == 1) & (test_df['is_fraud'] == 0)
# ]
# false_negatives = test_df[
#     (test_df['pred'] == 0) & (test_df['is_fraud'] == 1)
# ]
# if len(false_negatives) == 0:
#     print("✅ No false negatives found (recall = 1.0)")
# else:
#     false_negatives[['message_text']].sample(
#         min(5, len(false_negatives)), random_state=42
#     )

#     false_negatives[['message_text']].sample(5, random_state=42)
