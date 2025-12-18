from email.mime import text
import pandas as pd
import re
import os

# -----------------------------
# Constants
# -----------------------------

OFFICIAL_DOMAINS = [
    "safaricom.co.ke",
    "mpesa.co.ke",
    "m-pesa.com"
]

SHORTENERS = [
    "bit.ly",
    "tinyurl.com"
]

URGENT_WORDS = [
    "urgent", "immediately", "verify", "confirm",
    "suspended", "blocked", "unlock", "security",
    "alert", "claim", "winner", "prize"
]

SPELLING_ERRORS = [
    "confimed", "payed", "balanse",
    "ballance", "trasaction", "transction"
]

# -----------------------------
# Helper functions
# -----------------------------

def find_links(text):
    pattern = r'(http[s]?://[^\s]+|www\.[^\s]+)'
    return re.findall(pattern, text.lower())

def extract_link_features(text):
    features = {}

    links = find_links(text)
    text_lower = text.lower()

    features["has_link"] = int(len(links) > 0)
    features["link_count"] = len(links)

    features["has_shortened_link"] = int(
        any(short in link for link in links for short in SHORTENERS)
    )

    features["has_official_domain"] = int(
        any(domain in text_lower for domain in OFFICIAL_DOMAINS)
    )

    # Link position
    if links:
        first_pos = text_lower.find(links[0])
        ratio = first_pos / max(len(text_lower), 1)
        features["link_at_start"] = int(ratio < 0.3)
        features["link_at_end"] = int(ratio > 0.7)
    else:
        features["link_at_start"] = 0
        features["link_at_end"] = 0

    # Link + urgency context
    features["link_with_urgency"] = 0
    if links:
        for link in links:
            pos = text_lower.find(link)
            context = text_lower[max(0, pos-30):pos+30]
            if any(word in context for word in URGENT_WORDS):
                features["link_with_urgency"] = 1
                break

    # Link without transaction keywords
    has_transaction = ("confirmed" in text_lower) and ("ksh" in text_lower)
    features["link_without_transaction"] = int(
        features["has_link"] and not has_transaction
    )

    return features

# -----------------------------
# Main feature extractor
# -----------------------------

def extract_features(message_text, sender_id):
    features = {}

    text = str(message_text)
    text_lower = text.lower()

    # ---- Sender features ----
    
    sender_norm = str(sender_id).replace("-", "").upper()
    features["is_valid_sender"] = int(sender_norm == "MPESA" or sender_norm == "SAFARICOM")
    features["sender_is_numeric"] = int(sender_id.startswith("+254") or sender_id.isdigit())
    features["sender_length"] = len(sender_id)
        # Local Kenyan Pattern
    kenyan_pattern = r'\b(07|01|\+2547|\+2541)[0-9]{8}\b'
    # Global International Pattern (looks for + followed by country code)
    global_pattern = r'\+\d{1,3}\s?\d{4,12}'

    features['contains_local_number'] = int(bool(re.search(kenyan_pattern, text)))
    features['contains_foreign_number'] = int(bool(re.search(global_pattern, text)) and not bool(re.search(kenyan_pattern, text)))

    # ---- Text structure ----
    features["message_length"] = len(text)
    features["exclamation_count"] = text.count("!")
    features["has_confirmed"] = int("confirmed" in text_lower)
    is_confirmed_msg = "confirmed" in text_lower
    features["sender_id_mismatch"] = int(is_confirmed_msg and sender_norm != "MPESA")
    features["has_ksh"] = int("ksh" in text_lower)

    # ---- Amount features ----
    amounts = re.findall(r'Ksh\s?([\d,]+(?:\.\d+)?)', text)
    features["amount_count"] = len(amounts)

    # ---- Fraud language ----
    features["has_pin_word"] = int("pin" in text_lower)
    has_urgent_word = int(any(word in text_lower for word in URGENT_WORDS))
    features["urgent_without_transaction"] = int(has_urgent_word and not is_confirmed_msg)
    features["urgent_word_count"] = sum(text_lower.count(word) for word in URGENT_WORDS)
    features["has_spelling_error"] = int(any(err in text_lower for err in SPELLING_ERRORS))
    # 2. Panic Signals: Count exclamation marks
    features['exclamation_count'] = text.count('!')
    # 3. Urgency Screaming: Count words that are fully CAPITALIZED (excluding 'KSH')
    all_caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
    features['all_caps_count'] = len([w for w in all_caps_words if w != 'KSH'])
    words = text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 2 and w != 'KSH']
    features['all_caps_ratio'] = len(caps_words) / max(len(words), 1) if words else 0
    features['fake_confirmation'] = int("confirmed" in text_lower and not features['is_valid_sender'])

    # ---- Link features ----
    links = find_links(text)
    is_phishy_link = 0
    if len(links) > 0:
        # If there is a link, check if ANY of the official domains are inside it
        is_official = any(dom in links[0] for dom in OFFICIAL_DOMAINS)
        is_phishy_link = 1 if not is_official else 0
    features["is_phishy_link"] = is_phishy_link


    link_features = extract_link_features(text)
    features.update(link_features)
    return features

# -----------------------------
# Dataset-level processing
# -----------------------------

def process_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    print(f"🔧 Extracting features from {len(df):,} messages...")

    feature_rows = []
    for _, row in df.iterrows():
        feats = extract_features(row["message_text"], row["sender_id"])
        feature_rows.append(feats)

    features_df = pd.DataFrame(feature_rows)
    final_df = pd.concat([df, features_df], axis=1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    print(f"✅ Features saved to {output_csv}")
    return final_df

# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    process_dataset(
        input_csv="data/raw/mpesa_sms_messages.csv",
        output_csv="data/processed/mpesa_sms_features.csv"
    )
