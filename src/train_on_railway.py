import os
import pandas as pd
import joblib
from train_promotion_model import build_preprocessor, NUMERIC_FEATURES, TEXT_FEATURE
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def train_and_save():
    print("🚀 Starting Cloud Training...")
    
    # 1. Load your dataset (Ensure this CSV is in your GitHub!)
    data_path = "data/processed/mpesa_sms_features.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found in GitHub!")
        return
    
    df = pd.read_csv(data_path)
    X = df[NUMERIC_FEATURES + [TEXT_FEATURE]]
    y = df['is_fraud']

    # 2. Build the same Pipeline used in your research
    pipeline = Pipeline([
        ('prep', build_preprocessor()),
        ('clf', XGBClassifier())
    ])

    # 3. Fit the model
    print("🏋️ Training model on cloud data...")
    pipeline.fit(X, y)

    # 4. Create directory and Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/fraud_detector_pipeline.joblib')
    joblib.dump(pipeline, 'models/promotion_fraud_model.joblib')
    
    print("✅ Training Complete. Models saved to /app/models/")

if __name__ == "__main__":
    train_and_save()