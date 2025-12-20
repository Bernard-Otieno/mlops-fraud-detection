"""
PRODUCTION-GRADE M-PESA SMS Data Generator
Includes realistic fraud campaigns with sender pools, burst timing, and amount reuse
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

fake = Faker()
np.random.seed(42)
random.seed(42)

# ============================================================================
# FRAUD CAMPAIGN INFRASTRUCTURE
# ============================================================================

# Sender ID pools - fraud campaigns reuse small sets of numbers
FRAUD_SENDER_POOLS = {
    "smart_social": [],
    "smart_phish": [],
    "smart_reversal": [],
    "pin_request": [],
    "phishing_link": [],
    "reversal_scam": []
}

# Campaign timing - fraud comes in bursts
FRAUD_CAMPAIGN_TIME = {}

# Amount templates - scams reuse specific amounts
FRAUD_AMOUNTS = {
    "smart_social": [4999, 9999, 14999, 19999],
    "smart_phish": [2999, 7999, 12999],
    "smart_reversal": [5000, 10000, 15000],
    "pin_request": [1000, 5000, 10000],
    "phishing_link": [3000, 8000, 15000],
    "reversal_scam": [7500, 12500, 20000]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_fraud_sender():
    """Generate a new fraud sender number"""
    return "+2547" + "".join(str(random.randint(0, 9)) for _ in range(8))

def get_fraud_sender(fraud_type):
    """
    Get sender ID for fraud - 70% reuse within campaign, 30% new
    This creates realistic fraud patterns where same numbers are used repeatedly
    """
    pool = FRAUD_SENDER_POOLS.get(fraud_type, [])
    
    # 70% chance to reuse existing sender from this campaign
    if pool and random.random() < 0.7:
        sender_id = random.choice(pool)
    else:
        # 30% chance to introduce new sender
        sender_id = generate_fraud_sender()
        pool.append(sender_id)
        # Keep pool size reasonable (max 5 senders per campaign)
        if len(pool) > 5:
            pool.pop(0)
    
    return sender_id

def get_fraud_timestamp(fraud_type, base_time):
    """
    Generate timestamp for fraud with burst behavior
    Fraud comes in campaigns with messages clustered in time
    """
    # 20% chance to start a new campaign burst
    if fraud_type not in FRAUD_CAMPAIGN_TIME or random.random() < 0.2:
        FRAUD_CAMPAIGN_TIME[fraud_type] = base_time
    
    # Advance time by 1-5 minutes (burst behavior)
    delta = timedelta(minutes=random.randint(1, 5))
    FRAUD_CAMPAIGN_TIME[fraud_type] += delta
    return FRAUD_CAMPAIGN_TIME[fraud_type]

def get_fraud_amount(fraud_type):
    """
    Generate fraud amount with campaign signature
    Scams reuse specific amounts with small variations
    """
    if fraud_type in FRAUD_AMOUNTS:
        base = random.choice(FRAUD_AMOUNTS[fraud_type])
        # Add small noise to make it less obvious
        noise = random.randint(-50, 50)
        return round(base + noise, 2)
    else:
        # Fallback for other fraud types
        return round(random.uniform(1000, 100000), 2)

def get_transaction_cost(amount):
    """Calculate M-PESA transaction cost based on amount"""
    if amount <= 100:
        return 0
    elif amount <= 500:
        return 5
    elif amount <= 1000:
        return 10
    elif amount <= 1500:
        return 15
    elif amount <= 2500:
        return 20
    elif amount <= 3500:
        return 25
    elif amount <= 5000:
        return 30
    elif amount <= 7500:
        return 35
    elif amount <= 10000:
        return 40
    elif amount <= 15000:
        return 45
    elif amount <= 20000:
        return 50
    elif amount <= 35000:
        return 55
    elif amount <= 50000:
        return 60
    elif amount <= 70000:
        return 65
    elif amount <= 150000:
        return 105
    else:
        return 105

def generate_kenyan_name():
    """Generate realistic Kenyan names"""
    first_names = ['John', 'Jane', 'Peter', 'Mary', 'David', 'Sarah', 'James', 
                   'Grace', 'Daniel', 'Faith', 'Joseph', 'Ruth', 'Samuel', 'Esther',
                   'Michael', 'Lucy', 'Paul', 'Joyce', 'Timothy', 'Catherine',
                   'Kamau', 'Wanjiru', 'Otieno', 'Akinyi', 'Kipchoge', 'Chebet']
    
    last_names = ['Mwangi', 'Omondi', 'Kiplagat', 'Wanjiku', 'Kariuki', 'Otieno',
                  'Mutua', 'Kamau', 'Njoroge', 'Odhiambo', 'Kiprotich', 'Maina',
                  'Kimani', 'Owino', 'Koech', 'Nyambura', 'Waithaka', 'Ogola']
    
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_phone_number():
    """Generate Kenyan phone number"""
    prefixes = ['0710', '0720', '0730', '0740', '0750', '0768', '0769', '0791', '0792', '0793']
    number = random.choice(prefixes) + ''.join([str(random.randint(0, 9)) for _ in range(6)])
    return f"{number[:4]} {number[4:7]} {number[7:]}"

def format_mpesa_datetime(timestamp):
    """Format datetime in M-PESA format"""
    date_str = f"{timestamp.day}/{timestamp.month}/{timestamp.strftime('%y')}"
    time_str = timestamp.strftime("%I.%M %p").lstrip("0")
    return date_str, time_str

# ============================================================================
# LEGITIMATE MESSAGE GENERATORS
# ============================================================================

def generate_legitimate_payment_sms(msg_id):
    """Generate legitimate payment SMS with realistic promotions"""
    
    amount = round(random.uniform(50, 50000), 2)
    recipient = generate_kenyan_name()
    
    # Legitimate messages have evenly distributed timestamps
    days_ago = random.randint(0, 90)
    hours = random.randint(6, 22)
    minutes = random.randint(0, 59)
    timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
    
    date_str, time_str = format_mpesa_datetime(timestamp)
    
    # Calculate transaction cost
    cost = get_transaction_cost(amount)
    
    # Generate realistic balance
    old_balance = round(random.uniform(amount + cost + 100, 100000), 2)
    new_balance = round(old_balance - amount - cost, 2)
    
    # Daily limit
    daily_limit = 500000.00
    
    # Realistic promotions with links (60% have promotions, 40% of those have links)
    promotions_with_links = [
        "Pata zawadi za hadi sh 300 unapotumia M-PESA Global. Visit https://bit.ly/mpesalnk",
        "Lipa na M-PESA online na upate up to 25% off. Details at https://bit.ly/mpesaglobal",
        "Shop with M-PESA and get cashback. Check https://www.safaricom.co.ke/mpesa",
        "Exclusive M-PESA deals at https://bit.ly/safaricomdeals",
        "Send money globally with M-PESA. Learn more https://bit.ly/mpesalnk",
        "Get 10% cashback on all payments. Visit https://bit.ly/mpesacashback",
    ]
    
    promotions_no_links = [
        "Tumia M-PESA Biashara number kununua credit na data bila ya transaction cost.",
        "Lipa na M-PESA na ufaidike na bei nafuu.",
        "M-PESA Global - send money worldwide at low rates.",
        "Funga mkopo na M-PESA loans haraka.",
    ]
    
    # 60% chance of promotion, 40% of promotions have links
    if random.random() < 0.6:
        if random.random() < 0.4:
            promotion = random.choice(promotions_with_links)
        else:
            promotion = random.choice(promotions_no_links)
    else:
        promotion = ""
    
    # Build legitimate SMS
    sms = f"Confirmed. Ksh{amount:.2f} paid to {recipient}. on {date_str} at {time_str} New M-PESA balance is Ksh{new_balance:.2f}. Transaction cost, Ksh{cost:.2f}. Amount you can transact within the day is {daily_limit:.2f}."
    
    if promotion:
        sms += f" {promotion}"
    
    return {
        'message_id': f'MSG{msg_id:08d}',
        'sender_id': 'MPESA',
        'message_text': sms,
        'timestamp': timestamp,
        'amount': amount,
        'transaction_cost': cost,
        'new_balance': new_balance,
        'recipient_name': recipient,
        'is_fraud': 0,
        'fraud_type': None,
        'message_type': 'payment_confirmation'
    }

def generate_legitimate_receipt_sms(msg_id):
    """Generate legitimate receipt SMS"""
    
    amount = round(random.uniform(50, 50000), 2)
    sender_name = generate_kenyan_name()
    sender_number = generate_phone_number()
    
    # Evenly distributed timestamps for legit messages
    days_ago = random.randint(0, 90)
    hours = random.randint(6, 22)
    minutes = random.randint(0, 59)
    timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
    
    date_str, time_str = format_mpesa_datetime(timestamp)
    
    old_balance = round(random.uniform(100, 80000), 2)
    new_balance = round(old_balance + amount, 2)
    
    # Promotions for receipts (less common, 30% chance)
    promotions = [
        "Lipa na M-PESA at NAIVAS SUPERMARKET and stand a chance to win a trolley shopping.",
        "Get exclusive deals at https://bit.ly/mpesadeals for M-PESA users.",
        "Save more with M-PESA. Visit https://www.safaricom.co.ke/offers",
        "Enjoy up to 25% off when you pay with Lipa na M-PESA at selected outlets.",
        "Pay with M-PESA today and stand a chance to win exciting rewards.",
        ""
    ]
    
    if random.random() < 0.30:
        promotion = random.choice(promotions)
    else:
        promotion = ""
    
    sms = f"Confirmed. You have received Ksh{amount:.2f} from {sender_name} {sender_number} on {date_str} at {time_str} New M-PESA balance is Ksh{new_balance:.2f}."
    
    if promotion:
        sms += f" {promotion}"
    
    return {
        'message_id': f'MSG{msg_id:08d}',
        'sender_id': 'MPESA',
        'message_text': sms,
        'timestamp': timestamp,
        'amount': amount,
        'transaction_cost': 0,
        'new_balance': new_balance,
        'sender_name': sender_name,
        'sender_number': sender_number,
        'is_fraud': 0,
        'fraud_type': None,
        'message_type': 'receipt_confirmation'
    }

def generate_legitimate_system_alert(msg_id):
    """Generate legitimate system alerts (PIN reminders, maintenance notices)"""
    
    LEGIT_PIN_MESSAGES = [
        "Safaricom Alert: Never share your M-PESA PIN with anyone, including Safaricom staff.",
        "Security Tip: Protect your account. Change your M-PESA PIN regularly.",
        "Reminder: Do not disclose your PIN to anyone claiming to be from Safaricom.",
        "M-PESA Security: Safaricom will only call you through 0722 000 000.",
    ]
    
    LEGIT_URGENT_ALERTS = [
        "URGENT: M-PESA service will be temporarily unavailable tonight from 12:00 AM to 2:00 AM due to maintenance.",
        "IMPORTANT: Scheduled M-PESA maintenance tonight. Some services may be affected.",
        "NOTICE: Safaricom is upgrading M-PESA systems to serve you better.",
        "ALERT: M-PESA system maintenance is underway. We apologize for any inconvenience.",
    ]
    
    # Pick category
    if random.random() < 0.5:
        sms = random.choice(LEGIT_PIN_MESSAGES)
        msg_type = "security_reminder"
    else:
        sms = random.choice(LEGIT_URGENT_ALERTS)
        msg_type = "system_maintenance"
    
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
    
    return {
        'message_id': f'MSG{msg_id:08d}',
        'sender_id': 'Safaricom',
        'message_text': sms,
        'timestamp': timestamp,
        'amount': 0.0,
        'transaction_cost': 0.0,
        'new_balance': 0.0,
        'is_fraud': 0,
        'fraud_type': None,
        'message_type': msg_type
    }

# ============================================================================
# FRAUD MESSAGE GENERATORS
# ============================================================================

def generate_smart_social_fraud_sms(msg_id, base_time):
    """
    High-fidelity social engineering fraud
    Uses campaign sender pool and burst timing
    """
    # Use campaign infrastructure
    sender_id = get_fraud_sender('smart_social')
    timestamp = get_fraud_timestamp('smart_social', base_time)
    amount = get_fraud_amount('smart_social')
    
    date_str, time_str = format_mpesa_datetime(timestamp)
    
    sender_name = generate_kenyan_name()
    sender_number = generate_phone_number()
    
    transaction_cost = round(random.choice([0, 35, 55]), 2)
    new_balance = round(random.uniform(10000, 200000), 2)
    
    # Core legit-looking receipt
    sms = (
        f"Confirmed. You have received Ksh{amount:.2f} from "
        f"{sender_name} {sender_number} on {date_str} at {time_str}. "
        f"Transaction cost, Ksh{transaction_cost:.2f}. "
        f"New M-PESA balance is Ksh{new_balance:.2f}."
    )
    
    # Soft social engineering nudge
    followups = [
        " If this transaction was not expected, kindly review your M-PESA activity.",
        " For your security, please ensure your M-PESA details are up to date.",
        " If you do not recognize this transaction, please contact Safaricom support.",
        " Kindly verify this transaction in your M-PESA menu for confirmation."
    ]
    
    if random.random() < 0.7:
        sms += random.choice(followups)
    
    return {
        'message_id': f'MSG{msg_id:08d}',
        'sender_id': sender_id,
        'message_text': sms,
        'timestamp': timestamp,
        'amount': amount,
        'transaction_cost': transaction_cost,
        'new_balance': new_balance,
        'is_fraud': 1,
        'fraud_type': 'smart_social',
        'message_type': 'fraudulent'
    }

def generate_fraud_sms(msg_id, fraud_type, base_time):
    """Generate various types of fraudulent SMS with campaign behavior"""
    
    # Use campaign infrastructure for applicable fraud types
    if fraud_type in FRAUD_SENDER_POOLS:
        sender_id = get_fraud_sender(fraud_type)
        timestamp = get_fraud_timestamp(fraud_type, base_time)
        amount = get_fraud_amount(fraud_type)
    else:
        # Fallback for other fraud types
        fake_sender_ids = ['M-PESA', 'MPESA-KE', '+254700000000', 'SAFMPESA', '20060']
        sender_id = random.choice(fake_sender_ids)
        timestamp = base_time
        amount = round(random.uniform(1000, 100000), 2)
    
    date_str, time_str = format_mpesa_datetime(timestamp)
    
    if fraud_type == 'fake_sender_id':
        recipient = generate_kenyan_name()
        cost = get_transaction_cost(amount)
        new_balance = round(random.uniform(1000, 50000), 2)
        
        sms = f"Confirmed. Ksh{amount:.2f} paid to {recipient}. on {date_str} at {time_str} New M-PESA balance is Ksh{new_balance:.2f}. Transaction cost, Ksh{cost:.2f}."
    
    elif fraud_type == 'spelling_errors':
        sender_id = 'MPESA'
        recipient = generate_kenyan_name()
        cost = get_transaction_cost(amount)
        new_balance = round(random.uniform(1000, 50000), 2)
        
        errors = [
            f"Confimed. Ksh{amount:.2f} paid to {recipient}. on {date_str} at {time_str} New M-PESA balance is Ksh{new_balance:.2f}. Trasaction cost, Ksh{cost:.2f}.",
            f"Confirmed. Ksh{amount:.2f} payed to {recipient}. on {date_str} at {time_str} New M-PESA balanse is Ksh{new_balance:.2f}. Transaction cost, Ksh{cost:.2f}.",
            f"Confirmed. Ksh{amount:.2f} paid to {recipient}. on {date_str} at {time_str} New M-PESA ballance is Ksh{new_balance:.2f}. Transction cost, Ksh{cost:.2f}."
        ]
        sms = random.choice(errors)
    
    elif fraud_type == 'reversal_scam':
        sender_name = generate_kenyan_name()
        sender_number = generate_phone_number()
        new_balance = round(random.uniform(amount + 1000, 100000), 2)
        
        sms = f"Confirmed. You have received Ksh{amount:.2f} from {sender_name} {sender_number} on {date_str} at {time_str} New M-PESA balance is Ksh{new_balance:.2f}. URGENT: This was sent by mistake. Please reverse immediately or call {sender_number}."
    
    elif fraud_type == 'pin_request':
        messages = [
            f"MPESA ALERT: Your account will be suspended. To verify, reply with your PIN and ID number. Reference: {random.randint(100000, 999999)}",
            f"M-PESA Security: Suspicious activity detected. Confirm your identity by sending PIN to this number within 24 hours.",
            f"URGENT: Your M-PESA account requires verification. Send your PIN and full names to +254{random.randint(700000000, 799999999)} to avoid suspension."
        ]
        sms = random.choice(messages)
    
    elif fraud_type == 'phishing_link':
        fake_links = [
            'https://bit.ly/mpesa-verify123',
            'https://bit.ly/urgent-mpesa',
            'https://tinyurl.com/mpesa-unlock',
            'http://mpesa-verify.com/account',
            'https://bit.ly/M-PESA-security',
        ]
        link = random.choice(fake_links)
        
        messages = [
            f"MPESA ALERT: Your account will be suspended. Verify immediately at {link}",
            f"URGENT: Suspicious activity detected on your M-PESA. Confirm here: {link}",
            f"M-PESA Security: Click {link} to verify your account within 24 hours.",
            f"Your M-PESA daily limit has been exceeded. Reset here: {link}",
        ]
        sms = random.choice(messages)
    
    elif fraud_type == 'smart_social':
        return generate_smart_social_fraud_sms(msg_id, base_time)
    
    else:  # grammar_errors, wrong_calculation, unrealistic_amount, etc.
        sender_id = 'MPESA'
        recipient = generate_kenyan_name()
        
        bad_grammar = [
            f"You has received Ksh{amount:.2f} from payment. Balance now Ksh{random.uniform(1000, 50000):.2f} is.",
            f"Confirmed money Ksh{amount:.2f} to {recipient} sent. New balance Ksh{random.uniform(1000, 50000):.2f}.",
        ]
        sms = random.choice(bad_grammar)
    
    # Add realistic padding to 75% of fraud messages
    if random.random() < 0.75:
        padding_options = [
            " This is an automated message from Safaricom.",
            " For more details, visit your nearest M-PESA agent.",
            " Thank you for using M-PESA services.",
        ]
        sms += random.choice(padding_options)
    
    return {
        'message_id': f'MSG{msg_id:08d}',
        'sender_id': sender_id,
        'message_text': sms,
        'timestamp': timestamp,
        'amount': amount,
        'transaction_cost': None,
        'new_balance': None,
        'is_fraud': 1,
        'fraud_type': fraud_type,
        'message_type': 'fraudulent'
    }

# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_dataset(n_samples=10000):
    """Generate complete SMS dataset with realistic fraud campaigns"""
    
    print("🚀 Generating PRODUCTION-GRADE M-PESA SMS Dataset...")
    print(f"📊 Target samples: {n_samples:,}\n")
    
    messages = []
    
    # 70% legitimate (split between payments, receipts, system alerts)
    n_legitimate = int(n_samples * 0.70)
    n_system_alerts = int(n_legitimate * 0.10)
    n_legitimate_transactions = n_legitimate - n_system_alerts
    n_payment = int(n_legitimate_transactions * 0.625)
    n_receipt = n_legitimate_transactions - n_payment
    
    print(f"✅ Generating {n_payment:,} legitimate payment messages...")
    for i in range(n_payment):
        messages.append(generate_legitimate_payment_sms(i))
        if (i + 1) % 1000 == 0:
            print(f"   Progress: {i+1:,}/{n_payment:,}")
    
    print(f"✅ Generating {n_receipt:,} legitimate receipt messages...")
    for i in range(n_payment, n_payment + n_receipt):
        messages.append(generate_legitimate_receipt_sms(i))
    
    print(f"✅ Generating {n_system_alerts:,} legitimate system alerts...")
    for i in range(n_payment + n_receipt, n_payment + n_receipt + n_system_alerts):
        messages.append(generate_legitimate_system_alert(i))
    
    # 30% fraudulent with campaign behavior
    n_fraud = n_samples - n_legitimate
    fraud_types = [
        'fake_sender_id',
        'spelling_errors', 
        'reversal_scam',
        'pin_request',
        'phishing_link',
        'smart_social',
        'grammar_errors'
    ]
    
    print(f"🚨 Generating {n_fraud:,} fraudulent messages with campaign behavior...")
    fraud_per_type = n_fraud // len(fraud_types)
    
    msg_id = n_payment + n_receipt + n_system_alerts
    base_time = datetime.now() - timedelta(days=90)
    
    for fraud_type in fraud_types:
        for _ in range(fraud_per_type):
            messages.append(generate_fraud_sms(msg_id, fraud_type, base_time))
            msg_id += 1
    
    # Add remaining
    while len(messages) < n_samples:
        fraud_type = random.choice(fraud_types)
        messages.append(generate_fraud_sms(msg_id, fraud_type, base_time))
        msg_id += 1
    
    df = pd.DataFrame(messages)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add 2% label noise to simulate real-world ambiguity
    noise_idx = df.sample(frac=0.02, random_state=42).index
    df.loc[noise_idx, 'is_fraud'] = 1 - df.loc[noise_idx, 'is_fraud']
    
    return df

def main():
    """Main execution"""
    
    df = generate_dataset(n_samples=10000)
    
    # Statistics
    print("\n" + "="*70)
    print("📈 DATASET STATISTICS")
    print("="*70)
    print(f"\n✅ Total Messages: {len(df):,}")
    print(f"🚨 Fraudulent: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"✔️  Legitimate: {(~df['is_fraud'].astype(bool)).sum():,} ({(~df['is_fraud'].astype(bool)).mean()*100:.1f}%)")
    
    print(f"\n📱 Message Types:")
    print(df['message_type'].value_counts().to_string())
    
    print(f"\n🎭 Fraud Types Distribution:")
    fraud_types = df[df['is_fraud']==1]['fraud_type'].value_counts()
    print(fraud_types.to_string())
    
    print(f"\n📤 Sender ID Distribution:")
    print(df['sender_id'].value_counts().head(10).to_string())
    
    # Check campaign behavior
    print(f"\n🔗 LINK ANALYSIS:")
    legit_with_links = df[(df['is_fraud']==0) & (df['message_text'].str.contains('http|www', case=False))].shape[0]
    fraud_with_links = df[(df['is_fraud']==1) & (df['message_text'].str.contains('http|www', case=False))].shape[0]
    print(f"   Legitimate messages with links: {legit_with_links:,}")
    print(f"   Fraudulent messages with links: {fraud_with_links:,}")
    
    # Check fraud sender reuse
    print(f"\n📞 FRAUD SENDER ANALYSIS:")
    fraud_senders = df[df['is_fraud']==1]['sender_id'].value_counts()
    print(f"   Unique fraud sender IDs: {len(fraud_senders)}")
    print(f"   Most reused fraud sender: {fraud_senders.iloc[0]} times")
    print(f"   Avg reuse per fraud sender: {fraud_senders.mean():.1f} times")
    
    # Save
    output_path = 'data/raw/mpesa_sms_messages.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset saved to: {output_path}")
    
    # Show samples
    print(f"\n🔍 Sample Messages:\n")
    print("="*70)
    print("LEGITIMATE PAYMENT:")
    print("="*70)
    legit_payment = df[(df['is_fraud']==0) & (df['message_type']=='payment_confirmation')].iloc[0]
    print(f"Sender: {legit_payment['sender_id']}")
    print(f"Message: {legit_payment['message_text']}\n")
    
    print("="*70)
    print("FRAUD WITH CAMPAIGN BEHAVIOR (smart_social):")
    print("="*70)
    fraud_sample = df[df['fraud_type']=='smart_social'].head(2)
    for idx, row in fraud_sample.iterrows():
        print(f"Sender: {row['sender_id']}")
        print(f"Timestamp: {row['timestamp']}")
        print(f"Amount: {row['amount']}")
        print(f"Message: {row['message_text'][:100]}...")
        print()
    
    print("="*70)
    print("✨ Dataset generation complete!")
    print("="*70)

if __name__ == "__main__":
    main()