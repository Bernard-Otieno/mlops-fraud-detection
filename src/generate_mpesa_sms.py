"""
IMPROVED M-PESA Transaction Data Generator
Now includes realistic promotional links in legitimate messages
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

fake = Faker()
np.random.seed(42)
random.seed(42)

# M-PESA transaction cost structure
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

def generate_legitimate_payment_sms(msg_id):
    """Generate legitimate payment SMS with realistic promotions"""
    
    amount = round(random.uniform(50, 50000), 2)
    recipient = generate_kenyan_name()
    
    # Generate timestamp
    days_ago = random.randint(0, 90)
    hours = random.randint(6, 22)
    minutes = random.randint(0, 59)
    timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
    
    date_str = f"{timestamp.day}/{timestamp.month}/{timestamp.strftime('%y')}"
    time_str = timestamp.strftime("%I.%M %p").lstrip("0")
    
    # Calculate transaction cost
    cost = get_transaction_cost(amount)
    
    # Generate realistic balance
    old_balance = round(random.uniform(amount + cost + 100, 100000), 2)
    new_balance = round(old_balance - amount - cost, 2)
    
    # Daily limit
    daily_limit = 500000.00
    
    # UPDATED: Realistic promotions with links (60% have promotions, 40% of those have links)
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
    if random.random() < 0.6:  # 60% have promotion
        if random.random() < 0.4:  # 40% of promotions have links
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
    
    days_ago = random.randint(0, 90)
    hours = random.randint(6, 22)
    minutes = random.randint(0, 59)
    timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
    
    date_str = f"{timestamp.day}/{timestamp.month}/{timestamp.strftime('%y')}"
    time_str = timestamp.strftime("%I.%M %p").lstrip("0")
    
    old_balance = round(random.uniform(100, 80000), 2)
    new_balance = round(old_balance + amount, 2)
    
    # Promotions for receipts (less common, 30% chance)
    promotions = [
        "Lipa na M-PESA at NAIVAS SUPERMARKET and stand a chance to win a trolley shopping.",
        "Get exclusive deals at https://bit.ly/mpesadeals for M-PESA users.",
        "Save more with M-PESA. Visit https://www.safaricom.co.ke/offers",
        ""
    ]
    
    if random.random() < 0.3:  # 30% have promotion
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

def generate_fraud_sms(msg_id, fraud_type):
    """Generate various types of fraudulent SMS"""
    
    amount = round(random.uniform(1000, 100000), 2)
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
    
    if fraud_type == 'fake_sender_id':
        fake_sender_ids = ['M-PESA', 'MPESA-KE', '+254700000000', 'SAFMPESA', '20060']
        sender_id = random.choice(fake_sender_ids)
        
        recipient = generate_kenyan_name()
        date_str = f"{timestamp.day}/{timestamp.month}/{timestamp.strftime('%y')}"
        time_str = timestamp.strftime("%I.%M %p").lstrip("0")
        cost = get_transaction_cost(amount)
        new_balance = round(random.uniform(1000, 50000), 2)
        
        sms = f"Confirmed. Ksh{amount:.2f} paid to {recipient}. on {date_str} at {time_str} New M-PESA balance is Ksh{new_balance:.2f}. Transaction cost, Ksh{cost:.2f}."
    
    elif fraud_type == 'spelling_errors':
        sender_id = 'MPESA'
        recipient = generate_kenyan_name()
        date_str = f"{timestamp.day}/{timestamp.month}/{timestamp.strftime('%y')}"
        time_str = timestamp.strftime("%I.%M %p").lstrip("0")
        cost = get_transaction_cost(amount)
        new_balance = round(random.uniform(1000, 50000), 2)
        
        errors = [
            f"Confimed. Ksh{amount:.2f} paid to {recipient}. on {date_str} at {time_str} New M-PESA balance is Ksh{new_balance:.2f}. Trasaction cost, Ksh{cost:.2f}.",
            f"Confirmed. Ksh{amount:.2f} payed to {recipient}. on {date_str} at {time_str} New M-PESA balanse is Ksh{new_balance:.2f}. Transaction cost, Ksh{cost:.2f}.",
            f"Confirmed. Ksh{amount:.2f} paid to {recipient}. on {date_str} at {time_str} New M-PESA ballance is Ksh{new_balance:.2f}. Transction cost, Ksh{cost:.2f}."
        ]
        sms = random.choice(errors)
    
    elif fraud_type == 'reversal_scam':
        sender_id = random.choice(['MPESA', 'M-PESA', '+254700000000'])
        sender_name = generate_kenyan_name()
        sender_number = generate_phone_number()
        date_str = f"{timestamp.day}/{timestamp.month}/{timestamp.strftime('%y')}"
        time_str = timestamp.strftime("%I.%M %p").lstrip("0")
        new_balance = round(random.uniform(amount + 1000, 100000), 2)
        
        sms = f"Confirmed. You have received Ksh{amount:.2f} from {sender_name} {sender_number} on {date_str} at {time_str} New M-PESA balance is Ksh{new_balance:.2f}. URGENT: This was sent by mistake. Please reverse immediately or call {sender_number}."
    
    elif fraud_type == 'pin_request':
        sender_id = random.choice(['MPESA', 'M-PESA', 'SAFARICOM', '+254700000000'])
        
        messages = [
            f"MPESA ALERT: Your account will be suspended. To verify, reply with your PIN and ID number. Reference: {random.randint(100000, 999999)}",
            f"M-PESA Security: Suspicious activity detected. Confirm your identity by sending PIN to this number within 24 hours.",
            f"URGENT: Your M-PESA account requires verification. Send your PIN and full names to +254{random.randint(700000000, 799999999)} to avoid suspension."
        ]
        sms = random.choice(messages)
    
    elif fraud_type == 'phishing_link':
        sender_id = random.choice(['MPESA', 'M-PESA', 'SAFMPESA', '+254700000000'])
        
        # UPDATED: More realistic phishing links
        fake_links = [
            'https://bit.ly/mpesa-verify123',
            'https://bit.ly/urgent-mpesa',
            'https://tinyurl.com/mpesa-unlock',
            'http://mpesa-verify.com/account',
            'https://bit.ly/M-PESA-security',
            'https://bit.ly/mpesa-suspended',
            'http://safaricom-verify.co.ke/mpesa',
        ]
        link = random.choice(fake_links)
        
        messages = [
            f"MPESA ALERT: Your account will be suspended. Verify immediately at {link}",
            f"URGENT: Suspicious activity detected on your M-PESA. Confirm here: {link}",
            f"M-PESA Security: Click {link} to verify your account within 24 hours or lose access.",
            f"Your M-PESA daily limit has been exceeded. Reset here: {link}",
            f"CONGRATULATIONS! You won Ksh{amount:.2f} in M-PESA promotion. Claim at {link}",
            f"Your account has been locked due to suspicious activity. Unlock at {link}",
        ]
        sms = random.choice(messages)
    
    elif fraud_type == 'wrong_calculation':
        sender_id = 'MPESA'
        recipient = generate_kenyan_name()
        date_str = f"{timestamp.day}/{timestamp.month}/{timestamp.strftime('%y')}"
        time_str = timestamp.strftime("%I.%M %p").lstrip("0")
        cost = get_transaction_cost(amount)
        
        old_balance = round(random.uniform(1000, 50000), 2)
        wrong_new_balance = round(old_balance + random.uniform(-5000, 5000), 2)
        
        sms = f"Confirmed. Ksh{amount:.2f} paid to {recipient}. on {date_str} at {time_str} New M-PESA balance is Ksh{wrong_new_balance:.2f}. Transaction cost, Ksh{cost:.2f}."
    
    elif fraud_type == 'unrealistic_amount':
        sender_id = random.choice(['MPESA', 'M-PESA'])
        sender_name = generate_kenyan_name()
        sender_number = generate_phone_number()
        
        huge_amount = round(random.uniform(500000, 10000000), 2)
        date_str = f"{timestamp.day}/{timestamp.month}/{timestamp.strftime('%y')}"
        time_str = timestamp.strftime("%I.%M %p").lstrip("0")
        new_balance = round(random.uniform(1000000, 15000000), 2)
        
        sms = f"Confirmed. You have received Ksh{huge_amount:.2f} from {sender_name} {sender_number} on {date_str} at {time_str} New M-PESA balance is Ksh{new_balance:.2f}."
    elif fraud_type == 'smart_reversal':
        # Perfectly formatted receipt from the 'Official' sender
        sender_id = 'MPESA'
        sender_name = generate_kenyan_name()
        sender_number = generate_phone_number()
        date_str = f"{timestamp.day}/{timestamp.month}/{timestamp.strftime('%y')}"
        time_str = timestamp.strftime("%I.%M %p").lstrip("0")
        sms = f"Confirmed. You have received Ksh{amount:.2f} from {sender_name} {sender_number} on {date_str} at {time_str} New M-PESA balance is Ksh{random.uniform(5000, 20000):.2f}."

    elif fraud_type == 'smart_phish':
        # Uses a URL that looks like the real Safaricom domain
        sender_id = 'MPESA'
        fake_domain = random.choice(["safaricom-portal.co.ke", "mpesa-security-update.com", "safaricom-care.net"])
        sms = f"M-PESA Alert: Suspicious activity detected on your account. To secure your funds, please verify your identity at https://{fake_domain}/verify."

    elif fraud_type == 'smart_social':
        # High-pressure social engineering with no links or errors
        sender_id = 'MPESA'
        sms = "IMPORTANT: Due to updated Safaricom regulations, your M-PESA PIN will expire in 24 hours. To prevent account suspension, call our security desk on 0700000000 immediately."

    else:  # grammar_errors
        sender_id = 'MPESA'
        recipient = generate_kenyan_name()
        
        bad_grammar = [
            f"You has received Ksh{amount:.2f} from payment. Balance now Ksh{random.uniform(1000, 50000):.2f} is.",
            f"Confirmed money Ksh{amount:.2f} to {recipient} sent. New balance Ksh{random.uniform(1000, 50000):.2f}.",
            f"Transaction confirmed of Ksh{amount:.2f}. The balance for M-PESA is now Ksh{random.uniform(1000, 50000):.2f} after deduction."
        ]
        sms = random.choice(bad_grammar)
    
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

def generate_dataset(n_samples=10000):
    """Generate complete SMS dataset"""
    
    print("🚀 Generating IMPROVED M-PESA SMS Dataset...")
    print(f"📊 Target samples: {n_samples:,}\n")
    
    messages = []
    
    # 80% legitimate (50% payment, 30% receipt)
    n_legitimate = int(n_samples * 0.80)
    n_payment = int(n_legitimate * 0.625)
    n_receipt = n_legitimate - n_payment
    
    print(f"✅ Generating {n_payment:,} legitimate payment messages (with promotional links)...")
    for i in range(n_payment):
        messages.append(generate_legitimate_payment_sms(i))
        if (i + 1) % 1000 == 0:
            print(f"   Progress: {i+1:,}/{n_payment:,}")
    
    print(f"✅ Generating {n_receipt:,} legitimate receipt messages...")
    for i in range(n_payment, n_payment + n_receipt):
        messages.append(generate_legitimate_receipt_sms(i))
        if (i + 1 - n_payment) % 1000 == 0:
            print(f"   Progress: {i+1-n_payment:,}/{n_receipt:,}")
    
    # 20% fraudulent
    n_fraud = n_samples - n_legitimate
    fraud_types = [
        'fake_sender_id',
        'spelling_errors', 
        'reversal_scam',
        'pin_request',
        'phishing_link',
        'wrong_calculation',
        'unrealistic_amount',
        'grammar_errors'
    ]
    
    print(f"🚨 Generating {n_fraud:,} fraudulent messages (with phishing links)...")
    fraud_per_type = n_fraud // len(fraud_types)
    
    msg_id = n_payment + n_receipt
    for fraud_type in fraud_types:
        for _ in range(fraud_per_type):
            messages.append(generate_fraud_sms(msg_id, fraud_type))
            msg_id += 1
    
    # Add remaining
    while len(messages) < n_samples:
        fraud_type = random.choice(fraud_types)
        messages.append(generate_fraud_sms(msg_id, fraud_type))
        msg_id += 1
    
    df = pd.DataFrame(messages)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
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
    
    # Check links in legitimate vs fraud
    print(f"\n🔗 LINK ANALYSIS:")
    legit_with_links = df[(df['is_fraud']==0) & (df['message_text'].str.contains('http|www', case=False))].shape[0]
    fraud_with_links = df[(df['is_fraud']==1) & (df['message_text'].str.contains('http|www', case=False))].shape[0]
    print(f"   Legitimate messages with links: {legit_with_links:,}")
    print(f"   Fraudulent messages with links: {fraud_with_links:,}")
    
    # Save
    output_path = 'data/raw/mpesa_sms_messages.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset saved to: {output_path}")
    
    # Show samples
    print(f"\n🔍 Sample Messages:\n")
    print("="*70)
    print("LEGITIMATE PAYMENT WITH LINK:")
    print("="*70)
    legit_with_link = df[(df['is_fraud']==0) & (df['message_text'].str.contains('http|bit.ly', case=False))]
    if len(legit_with_link) > 0:
        sample = legit_with_link.iloc[0]
        print(f"Sender: {sample['sender_id']}")
        print(f"Message: {sample['message_text']}\n")
    
    print("="*70)
    print("FRAUD WITH PHISHING LINK:")
    print("="*70)
    fraud_sample = df[df['fraud_type']=='phishing_link'].iloc[0]
    print(f"Sender: {fraud_sample['sender_id']}")
    print(f"Message: {fraud_sample['message_text']}\n")
    
    print("="*70)
    print("✨ Dataset generation complete!")
    print("="*70)

if __name__ == "__main__":
    main()