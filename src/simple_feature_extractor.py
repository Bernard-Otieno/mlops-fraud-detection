import pandas as pd
import re

def extract_features(message_text, sender_id):
    """
    Extract 5 simple features from an SMS
    
    Returns a dictionary with features
    """
    features = {}

    features['is_valid_sender'] = sender_id == 'MPESA'
    features['message_length'] = len(message_text)
    features['has_pin_word'] = 1 if re.search(r'\bpin\b', message_text.lower()) else 0
    amounts =re.findall(r'\d+', message_text)
    features['amount_count'] = len(amounts)
    features['has_confirmed'] = 'confirmed' in message_text.lower()
    link_patterns = ['http', 'https', 'www.', 'bit.ly', 'tinyurl']
    features['has_link'] = 1 if any(pattern in message_text.lower() for pattern in link_patterns) else 0    
    features['exclamation_count'] = message_text.count('!')
    urgent_pattern = r'\b(urgent|immediately|asap|suspended|blocked|expire)\b'
    features['has_urgent_word'] = 1 if re.search(urgent_pattern, message_text.lower()) else 0
    

    return features

#test the function
# Test cases
# test_cases = [
#     ("Lucy Kiplagat sent money", False),  # Should NOT match
#     ("Send your PIN now", True),          # Should match
#     ("Enter PIN to confirm", True),       # Should match
#     ("Shopping at the mall", False),      # Should NOT match
#     ("Your pin code is required", True),  # Should match
# ]

# print("\n🧪 Testing PIN detection:")
# print("="*60)
# for text, expected in test_cases:
#     # Test with word boundary
#     result = bool(re.search(r'\bpin\b', text.lower()))
#     status = "✅" if result == expected else "❌"
#     print(f"{status} '{text[:30]}...' → {result} (expected {expected})")

def process_all_messages(csv_path):
    """Process all messages and extract features"""
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} messages...")
    
    # Store all features
    all_features = []
    
    # Loop through each message
    for idx, row in df.iterrows():
        # Extract features for this message
        features = extract_features(row['message_text'], row['sender_id'])
        all_features.append(features)
        
        # Progress update
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Combine with original data
    result = pd.concat([df, features_df], axis=1)
    
    return result

# Run it!
if __name__ == "__main__":
    result_df = process_all_messages('data/raw/mpesa_sms_messages.csv')
    
    # Save
    result_df.to_csv('data/processed/my_features.csv', index=False)
    print(f"\n✅ Saved {len(result_df)} messages with features!")
    
    # Show correlation with fraud
    print("\nFeature correlation with fraud:")
    feature_cols = ['is_valid_sender', 'message_length', 'has_pin_word', 
                    'amount_count', 'has_confirmed', 'has_link', 'exclamation_count', 'has_urgent_word']
    for col in feature_cols:
        corr = result_df[[col, 'is_fraud']].corr().iloc[0, 1]
        print(f"  {col:20s}: {corr:+.3f}")