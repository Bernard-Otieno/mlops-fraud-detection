"""
Feature Extraction for M-PESA Promotion Fraud Detection
Specialized features for detecting fraudulent promotional messages
"""

import pandas as pd
import numpy as np
import re
import os

from feature_extractor import extract_financial_legitimacy_features

# ============================================================================
# CONSTANTS - Based on Real Safaricom Patterns
# ============================================================================

# Legitimate Safaricom indicators
LEGIT_DOMAINS = [
    'safaricom.co.ke',
    'mpesa.co.ke',
    'm-pesa.com',
    'rebrand.ly'  # Safaricom uses this for link shortening
]

LEGIT_USSD_CODES = [
    r'\*334#',   # M-PESA menu
    r'\*444#',   # Bundles
    r'\*544#',   # Balance check
    r'\*555#',   # Customer care
    r'\*126#',   # Bonga points
    r'\*544\*\d+#'  # Balance variants like *544*100#
]

LEGIT_SMS_CODES = [
    '22444',  # Safaricom competitions
    '24101',  # Games
    '22884',  # Other services
    '22888',
    '21212'
]

SAFARICOM_BRANDS = [
    'nyakua',
    'bonga',
    'shangwe',
    'mwelekoni',
    'safaricom@25',
    'transforming lives',
    'lipa na m-pesa',
    'ziidi'
]

LEGIT_CONTACTS = [
    '100',   # Prepay customer care
    '200',   # Postpay customer care
    '234',   # M-PESA customer care
    '333'    # Corporate
]

# Fraud indicators
FRAUD_DOMAINS = [
    'bit.ly',
    'tinyurl.com',
    'goo.gl',
    't.co',
    'ow.ly'
]

FRAUD_KEYWORDS = [
    'pay.*fee',
    'send.*money',
    'processing.*fee',
    'registration.*fee',
    'activation.*fee',
    'delivery.*fee',
    'claim.*fee',
    'verification.*fee',
    'guaranteed.*win',
    'you.*won',
    'selected.*winner',
    'claim.*prize',
    'reply.*pin',
    'send.*pin'
]

URGENCY_KEYWORDS = [
    'urgent',
    'immediately',
    'hurry',
    'now',
    'expires',
    'expire',
    'limited time',
    'today only',
    'last chance',
    'act now',
    'fast',
    'quick',
    'asap',
    'limited slots',
    'running out',
    'don\'t miss',
    'ends soon',
    'ends tonight',
    'before.*expire'
]

WIN_CLAIM_WORDS = [
    'won',
    'winner',
    'congratulations',
    'selected',
    'lucky',
    'chosen',
    'qualified',
    'eligible'
]

GUARANTEE_PHRASES = [
    '100%',
    'guaranteed',
    'all win',
    'everyone wins',
    'everybody wins',
    'no catch',
    'risk free',
    'free gift',
    'instant win'
]

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_sender_features(sender_id):
    """Extract features from sender ID"""
    features = {}
    
    sender_lower = sender_id.lower()
    
    # Valid Safaricom senders
    valid_senders = ['safaricom', 'mpesa', 'm-pesa', 'bonga']
    features['is_legit_sender'] = int(any(s in sender_lower for s in valid_senders))
    
    # Suspicious sender patterns
    features['sender_is_number'] = int(sender_id.replace('+', '').replace('-', '').isdigit())
    features['sender_has_plus'] = int('+' in sender_id)
    features['sender_length'] = len(sender_id)
    
    # Impersonation attempts (e.g., "WINNR", "PROMO", "PRIZE")
    suspicious_senders = ['winner', 'winnr', 'promo', 'prize', 'bonus', 'reward', 'offer']
    features['suspicious_sender_name'] = int(any(s in sender_lower for s in suspicious_senders))
    
    return features


def extract_link_features(text):
    """Extract link-related features"""
    features = {}
    
    text_lower = text.lower()
    
    # Find all links
    link_pattern = r'http[s]?://[^\s]+|www\.[^\s]+'
    links = re.findall(link_pattern, text_lower)
    
    features['has_link'] = int(len(links) > 0)
    features['link_count'] = len(links)
    
    # Check for legitimate domains
    features['has_legit_domain'] = int(any(domain in text_lower for domain in LEGIT_DOMAINS))
    
    # Check for fraud link shorteners
    features['has_fraud_shortener'] = int(any(domain in text_lower for domain in FRAUD_DOMAINS))
    
    # Typosquatting detection (safaricom-xyz.com, mpesa-xyz.com)
    typosquat_patterns = [
        r'safaricom-[a-z]+\.com',
        r'mpesa-[a-z]+\.com',
        r'm-pesa-[a-z]+\.com',
        r'[a-z]+-safaricom\.',
        r'[a-z]+-mpesa\.'
    ]
    features['has_typosquat_domain'] = int(any(re.search(p, text_lower) for p in typosquat_patterns))
    
    # Link position (legit promos usually have links at end)
    if links:
        first_link_pos = text_lower.find(links[0])
        text_length = len(text_lower)
        features['link_position_ratio'] = first_link_pos / text_length if text_length > 0 else 0
        features['link_at_end'] = int(features['link_position_ratio'] > 0.7)
        features['link_at_start'] = int(features['link_position_ratio'] < 0.3)
    else:
        features['link_position_ratio'] = 0
        features['link_at_end'] = 0
        features['link_at_start'] = 0
    
    return features


def extract_contact_features(text):
    """Extract contact information features"""
    features = {}
    
    text_lower = text.lower()
    
    # USSD codes (legit indicator)
    ussd_count = sum(1 for code in LEGIT_USSD_CODES if re.search(code, text))
    features['has_ussd_code'] = int(ussd_count > 0)
    features['ussd_code_count'] = ussd_count
    
    # SMS short codes (legit indicator)
    sms_code_count = sum(1 for code in LEGIT_SMS_CODES if code in text)
    features['has_sms_shortcode'] = int(sms_code_count > 0)
    features['sms_shortcode_count'] = sms_code_count
    
    # Official contact numbers
    features['has_official_contact'] = int(any(contact in text for contact in LEGIT_CONTACTS))
    
    # Suspicious phone numbers (full mobile numbers in promo = red flag)
    mobile_pattern = r'0[17]\d{8}|\+254[17]\d{8}'
    mobile_matches = re.findall(mobile_pattern, text)
    features['has_mobile_number'] = int(len(mobile_matches) > 0)
    features['mobile_number_count'] = len(mobile_matches)
    
    # Paybill numbers (could be legit or fraud)
    paybill_pattern = r'paybill[:\s]*(\d+)'
    features['has_paybill'] = int(bool(re.search(paybill_pattern, text_lower)))
    
    return features


def extract_prize_features(text):
    """Extract prize/reward related features"""
    features = {}
    
    text_lower = text.lower()
    
    # Win claims
    features['win_claim_count'] = sum(1 for word in WIN_CLAIM_WORDS if word in text_lower)
    features['has_win_claim'] = int(features['win_claim_count'] > 0)
    
    # Extract prize amounts
    prize_patterns = [
        r'ksh\s*(\d+(?:,\d+)*(?:\.\d+)?)',
        r'(\d+(?:,\d+)*)\s*ksh',
        r'worth.*?ksh\s*(\d+(?:,\d+)*)',
    ]
    
    prize_amounts = []
    for pattern in prize_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                amount = float(match.replace(',', ''))
                prize_amounts.append(amount)
            except:
                pass
    
    features['has_prize_amount'] = int(len(prize_amounts) > 0)
    features['max_prize_amount'] = max(prize_amounts) if prize_amounts else 0
    features['prize_amount_count'] = len(prize_amounts)
    
    # Unrealistic prizes (>100K is suspicious for random SMS)
    features['unrealistic_prize'] = int(features['max_prize_amount'] > 100000)
    
    # Very large prizes (>500K is very suspicious)
    features['extreme_prize'] = int(features['max_prize_amount'] > 500000)
    
    # Guarantee phrases (100% guaranteed, everyone wins)
    features['guarantee_count'] = sum(1 for phrase in GUARANTEE_PHRASES if phrase in text_lower)
    features['has_guarantee'] = int(features['guarantee_count'] > 0)
    
    return features


def extract_payment_features(text):
    """Extract payment/fee request features"""
    features = {}
    
    text_lower = text.lower()
    
    # General payment requests
    payment_words = ['pay', 'send', 'transfer', 'deposit']
    features['has_payment_request'] = int(any(word in text_lower for word in payment_words))
    
    # Suspicious fee types
    fee_patterns = [
        'processing fee',
        'registration fee',
        'activation fee',
        'verification fee',
        'delivery fee',
        'handling fee',
        'claim fee',
        'withdrawal fee'
    ]
    features['suspicious_fee_count'] = sum(1 for pattern in fee_patterns if pattern in text_lower)
    features['has_suspicious_fee'] = int(features['suspicious_fee_count'] > 0)
    
    # "Pay X to claim Y" pattern (classic scam)
    pay_to_claim_pattern = r'(pay|send).*?(claim|receive|get|win)'
    features['pay_to_claim_pattern'] = int(bool(re.search(pay_to_claim_pattern, text_lower)))
    
    return features


def extract_urgency_features(text):
    """Extract urgency/pressure features"""
    features = {}
    
    text_lower = text.lower()
    
    # Urgency keywords
    features['urgency_count'] = sum(1 for word in URGENCY_KEYWORDS if word in text_lower)
    features['has_urgency'] = int(features['urgency_count'] > 0)
    features['high_urgency'] = int(features['urgency_count'] >= 2)
    
    # Time pressure phrases
    time_pressure = ['24 hours', '48 hours', '2 hours', 'today', 'tonight', 'now', 'immediately']
    features['has_time_pressure'] = int(any(phrase in text_lower for phrase in time_pressure))
    
    # Expiration mentions
    expire_pattern = r'expire[sd]?|expir(e|ing|ation)'
    features['mentions_expiration'] = int(bool(re.search(expire_pattern, text_lower)))
    
    return features


def extract_language_features(text):
    """Extract language quality features"""
    features = {}
    
    # Exclamation marks
    features['exclamation_count'] = text.count('!')
    features['exclamation_density'] = features['exclamation_count'] / len(text) if len(text) > 0 else 0
    features['excessive_exclamations'] = int(features['exclamation_count'] >= 3)
    
    # Question marks
    features['question_count'] = text.count('?')
    
    # ALL CAPS words (excluding MPESA, BONGA, etc.)
    all_caps_words = re.findall(r'\b[A-Z]{4,}\b', text)
    excluded_caps = ['MPESA', 'PESA', 'BONGA', 'FREE', 'PLAY']
    features['caps_word_count'] = len([w for w in all_caps_words if w not in excluded_caps])
    features['excessive_caps'] = int(features['caps_word_count'] >= 3)
    
    # Emojis (unusual in official Safaricom promos)
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF🎉🎊🎁🎈]'
    emoji_matches = re.findall(emoji_pattern, text)
    features['emoji_count'] = len(emoji_matches)
    features['has_emoji'] = int(features['emoji_count'] > 0)
    
    # Message length
    features['message_length'] = len(text)
    features['word_count'] = len(text.split())
    
    # Average word length (scams often use simpler language)
    words = text.split()
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    
    return features


def extract_legitimacy_features(text):
    """Extract features indicating legitimacy"""
    features = {}
    
    text_lower = text.lower()
    
    # Safaricom brand terms
    features['safaricom_brand_count'] = sum(1 for brand in SAFARICOM_BRANDS if brand in text_lower)
    features['has_safaricom_brand'] = int(features['safaricom_brand_count'] > 0)
    
    # Terms & Conditions mentions (legit promos usually have this)
    tc_phrases = ['t&c', 't&cs', 'terms and conditions', 'terms apply', 'conditions apply']
    features['mentions_terms'] = int(any(tc in text_lower for tc in tc_phrases))
    
    # "Stand a chance" vs "You won" (legit vs fraud)
    features['says_stand_a_chance'] = int('stand a chance' in text_lower)
    features['says_you_won'] = int(re.search(r'you (have )?won', text_lower) is not None)
    
    # Opt-out mentions (legit promos provide opt-out)
    features['mentions_optout'] = int('optout' in text_lower or 'opt out' in text_lower or 'opt-out' in text_lower)
    
    # Hashtags (common in real Safaricom promos)
    hashtag_count = len(re.findall(r'#\w+', text))
    features['hashtag_count'] = hashtag_count
    features['has_hashtag'] = int(hashtag_count > 0)
    
    # "Dial" for USSD (legit indicator)
    features['says_dial'] = int('dial' in text_lower)
    
    # SMS instructions (e.g., "SMS PLAY to 22444")
    features['has_sms_instruction'] = int(bool(re.search(r'sms \w+ to \d+', text_lower)))
    
    return features


def extract_scam_patterns(text):
    """Extract specific scam patterns"""
    features = {}
    
    text_lower = text.lower()
    
    # Click bait
    click_phrases = ['click here', 'click link', 'click to', 'tap here', 'tap link', 'click now']
    features['has_click_bait'] = int(any(phrase in text_lower for phrase in click_phrases))
    
    # Verification requests
    verify_words = ['verify', 'confirm', 'validate', 'authenticate']
    features['requests_verification'] = int(any(word in text_lower for word in verify_words))
    
    # Authority claims (CEO, official, board)
    authority_words = ['ceo', 'director', 'official', 'authorized', 'board', 'management']
    features['claims_authority'] = int(any(word in text_lower for word in authority_words))
    
    # Combined fraud patterns using regex
    fraud_pattern_count = sum(1 for pattern in FRAUD_KEYWORDS if re.search(pattern, text_lower))
    features['fraud_pattern_count'] = fraud_pattern_count
    features['has_fraud_pattern'] = int(fraud_pattern_count > 0)
    
    return features


def extract_composite_features(all_features):
    """Create composite features from existing ones"""
    composite = {}
    
    # High-risk combination: Win + Fee + Urgency
    composite['win_fee_urgency_combo'] = int(
        all_features.get('has_win_claim', 0) and
        all_features.get('has_suspicious_fee', 0) and
        all_features.get('high_urgency', 0)
    )
    
    # Suspicious link + Payment request
    composite['link_payment_combo'] = int(
        all_features.get('has_link', 0) and
        all_features.get('has_payment_request', 0)
    )
    
    # Unrealistic prize + Suspicious fee
    composite['unrealistic_prize_fee'] = int(
        all_features.get('unrealistic_prize', 0) and
        all_features.get('has_suspicious_fee', 0)
    )
    
    # No legitimate contacts + Payment request
    composite['no_legit_contact_payment'] = int(
        not all_features.get('has_official_contact', 0) and
        not all_features.get('has_ussd_code', 0) and
        all_features.get('has_payment_request', 0)
    )
    
    # Fraud link + Win claim
    composite['fraud_link_win_combo'] = int(
        all_features.get('has_fraud_shortener', 0) and
        all_features.get('has_win_claim', 0)
    )
    
    # Legitimacy score (0-6, higher = more legit)
    legitimacy_score = sum([
        all_features.get('is_legit_sender', 0),
        all_features.get('has_ussd_code', 0),
        all_features.get('has_sms_shortcode', 0),
        all_features.get('has_safaricom_brand', 0),
        all_features.get('mentions_terms', 0),
        all_features.get('has_legit_domain', 0)
    ])
    composite['legitimacy_score'] = legitimacy_score
    composite['high_legitimacy'] = int(legitimacy_score >= 3)
    
    # Fraud risk score (0-8, higher = more suspicious)
    fraud_score = sum([
        all_features.get('has_fraud_shortener', 0),
        all_features.get('has_suspicious_fee', 0),
        all_features.get('unrealistic_prize', 0),
        all_features.get('high_urgency', 0),
        all_features.get('has_guarantee', 0),
        all_features.get('suspicious_sender_name', 0),
        all_features.get('has_mobile_number', 0),
        all_features.get('says_you_won', 0)
    ])
    composite['fraud_risk_score'] = fraud_score
    composite['high_fraud_risk'] = int(fraud_score >= 4)
    
    return composite


def extract_all_features(message_text, sender_id):
    """Extract all features from a promotional message"""
    
    all_features = {}
    
    # Extract from different modules
    all_features.update(extract_sender_features(sender_id))
    all_features.update(extract_link_features(message_text))
    all_features.update(extract_contact_features(message_text))
    all_features.update(extract_prize_features(message_text))
    all_features.update(extract_payment_features(message_text))
    all_features.update(extract_urgency_features(message_text))
    all_features.update(extract_language_features(message_text))
    all_features.update(extract_legitimacy_features(message_text))
    all_features.update(extract_scam_patterns(message_text))
    financial_features = extract_financial_legitimacy_features(message_text, sender_id)
    all_features.update(financial_features)
    
    # Add composite features
    all_features.update(extract_composite_features(all_features))
    
    return all_features


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def process_dataset(input_csv, output_csv):
    """Process entire dataset and extract features"""
    
    df = pd.read_csv(input_csv)
    print(f"🔧 Extracting promotion features from {len(df):,} messages...")
    
    feature_rows = []
    for idx, row in df.iterrows():
        features = extract_all_features(row['message_text'], row['sender_id'])
        feature_rows.append(features)
        
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1:,}/{len(df):,}...")
    
    features_df = pd.DataFrame(feature_rows)
    final_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    print(f"✅ Features saved to {output_csv}")
    
    # Show feature summary
    print("\n" + "="*70)
    print("FEATURE SUMMARY")
    print("="*70)
    print(f"\nTotal features extracted: {len(features_df.columns)}")
    
    # Show correlation with fraud
    print("\n" + "="*70)
    print("TOP 15 FEATURES CORRELATED WITH FRAUD")
    print("="*70)
    
    correlations = final_df[list(features_df.columns) + ['is_fraud']].corr()['is_fraud'].sort_values(ascending=False)
    print(correlations[1:16].to_string())  # Top 15, excluding is_fraud itself
    
    return final_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    process_dataset(
        input_csv="data/raw/mpesa_promotion_messages.csv",
        output_csv="data/processed/mpesa_promotion_features.csv"
    )