from email.mime import text
import pandas as pd
import re
import os

from pygit2 import features

# -----------------------------
# Constants
# -----------------------------
AUTHORITY_PHRASES = [
    "safaricom",
    "customer care",
    "support",
    "help desk",
    "mpesa menu",
    "official"
]

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
    "urgent", "immediately", "now", "asap", "quickly", "verify", "confirm", 
    "blocked", "suspended", "locked", "freeze", "prize", "winner", "claim", 
    "reward", "security", "alert", "risk", "problem", "issue", "help"
]

SPELLING_ERRORS = [
    "confimed", "payed", "balanse",
    "ballance", "trasaction", "transction"
]

ACTION_VERBS = [
    'send', 'reply', 'click', 'call', 'visit',
    'confirm', 'verify', 'update', 'secure'
]
transaction_signals = [
    'confirmed',
    'ksh',
    'balance',
    'transaction cost',
    'new m-pesa balance'
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
    words = re.findall(r'\b\w+\b', text_lower)  # Simple word split
    num_words = len(words)
    text_length = len(text)

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
    features['message_length_ratio'] = len(text) / 160.0  # Relative to standard SMS length
    features["exclamation_count"] = text.count("!")
    features["has_confirmed"] = int("confirmed" in text_lower)
    is_confirmed_msg = "confirmed" in text_lower
    features["sender_id_mismatch"] = int(is_confirmed_msg and sender_norm != "MPESA")
    features["has_ksh"] = int("ksh" in text_lower)

    # ---- Amount features ----
    amounts = re.findall(r'Ksh\s?([\d,]+(?:\.\d+)?)', text)
    features["amount_count"] = len(amounts)

    

    # ---- Urgency features ----
    
    urgent_count = sum(1 for word in words if word in URGENT_WORDS)
    features["urgent_density"] = urgent_count / features["message_length"] if features["message_length"] > 0 else 0.0

    # Action verbs
    action_count = sum(text_lower.count(v) for v in ACTION_VERBS)
    features["action_verb_count"] = action_count/features["message_length"] if features["message_length"] > 0 else 0.0

    # transaction signals
    present = sum(1 for s in transaction_signals if s in text_lower)
    features['transaction_completeness'] = present / len(transaction_signals)


    # ---- Spelling errors ----
    features["has_spelling_error"] = int(any(err in text_lower for err in SPELLING_ERRORS))

    # 2. Panic Signals: Count exclamation marks
    features['exclamation_count'] = text.count('!')


    # 3. Urgency Screaming: Count words that are fully CAPITALIZED (excluding 'KSH')
    all_caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
    caps_count = sum(1 for c in message_text if c.isupper())

    features['all_caps_ratio'] = caps_count / features['message_length'] if features['message_length'] > 0 else 0.0
    features['exclamation_ratio'] = features['exclamation_ratio'] = (
    message_text.count('!') / features['message_length']
    ) if features['message_length'] > 0 else 0.0
    # 4. Fake Confirmation: Presence of 'confirmed
    features['fake_confirmation'] = int("confirmed" in text_lower and not features['is_valid_sender'])

    # 5. Authority Phrases: Presence of authority phrases
    features['authority_density'] = sum(1 for phrase in AUTHORITY_PHRASES if phrase in text_lower) / features['message_length'] if features['message_length'] > 0 else 0.0

    # ---- Link features ----
    links = find_links(text)
    features['link_count'] = len(links)
    features['has_shortener'] = int(any(short in link for link in links for short in ["bit.ly", "tinyurl.com,", "goo.gl", "ow.ly","t.co"]))
    features['link_with_urgency'] = 0
    if links:
        context = text_lower
        if any(word in context for word in URGENT_WORDS):
            features['link_with_urgency'] = 1
    
    features['sender_suspicious'] = int(sender_id.lower() in ["mpesa", "safaricom", "mpesa alert"] and not features['is_valid_sender'])
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
