"""
M-PESA Promotion Fraud Data Generator
Generates legitimate and fraudulent promotional messages
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
# LEGITIMATE PROMOTION GENERATORS
# ============================================================================

def generate_legitimate_promotion(msg_id):
    """Generate legitimate M-PESA/Safaricom promotion using real patterns"""
    
    timestamp = datetime.now() - timedelta(days=random.randint(0, 90))
    
    # Real Safaricom promotion patterns
    promo_templates = [
        # Anniversary/celebration promos (like Safaricom@25)
        {
            'text': "Tomorrow, your first call is on us! As we mark 25 years of transforming lives, enjoy 10 FREE minutes to connect, celebrate, and share the joy.",
            'category': 'anniversary_celebration',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': "Congratulations!! Your first call today is FREE for 10 minutes until midday! Thank you for being part of our amazing and impactful 25-year journey #Shangwe@25",
            'category': 'anniversary_celebration',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        
        # Game/competition promos
        {
            'text': "Today everybody is a winner! Ready to test your brainpower? Send the word PLAY to 24101 and grab FREE GIFTS!",
            'category': 'game_competition',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': f"You have {random.randint(5,20)} entries for Safaricom@25 Promo! Use M-PESA, Bonga, Buy bundles for a chance to win 1M! Dial *444*25#, *544*25#, *555*25# to Check Entries or OptOut",
            'category': 'loyalty_entries',
            'has_prize': True,
            'prize_amount': 1000000,
            'requires_fee': False,
            'has_terms': True
        },
        {
            'text': "Turn Bonga Points into endless fun! Stand a chance to win phones, airtime & more when you play cashless. Play now: https://rebrand.ly/f2955f",
            'category': 'bonga_promotion',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        
        # Data bonuses (Nyakua style)
        {
            'text': f"Enjoy {random.choice([50, 100, 200])} MB FREE Nyakua Bonus once you spend your daily target! You have used {random.randint(1,10)} MB so far, Spend {random.randint(20,90)} MB more today and get {random.choice([50, 100])} MB Free Data Bonus #MwelekeoniInternet.",
            'category': 'data_bonus',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': f"Congratulations! You have reached your target for today and have been awarded {random.choice([50, 100, 150])} MB Nyakua Bonus valid till midnight. For Balance Dial *544*100#",
            'category': 'data_bonus_achieved',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        
        # Partner promotions (Naivas, Carrefour pattern)
        {
            'text': f"Lipa na M-PESA at {random.choice(['Naivas', 'Carrefour', 'Quickmart', 'Chandarana'])} Supermarket this {random.choice(['week', 'weekend', 'month'])} and stand a chance to win shopping vouchers worth Ksh{random.choice([10000, 20000, 50000])}. Shop smart, pay smarter!",
            'category': 'partner_promo',
            'has_prize': True,
            'prize_amount': 20000,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': f"Get {random.choice([5, 10, 15])}% cashback when you pay with M-PESA at {random.choice(['Naivas', 'Carrefour', 'Java House', 'KFC'])} this weekend. Maximum cashback Ksh{random.choice([200, 500, 1000])}. Visit www.safaricom.co.ke/mpesa for T&Cs.",
            'category': 'cashback',
            'has_prize': True,
            'prize_amount': 500,
            'requires_fee': False,
            'has_terms': True
        },
        
        # Utility/bill payment cashback
        {
            'text': f"Pay your {random.choice(['electricity', 'water', 'KPLC', 'NHIF'])} bill via M-PESA and get {random.choice([5, 10])}% instant cashback! Minimum payment Ksh1000. Offer valid until {random.choice(['31st Dec', '31st Jan', 'month end'])}. Terms apply.",
            'category': 'utility_cashback',
            'has_prize': True,
            'prize_amount': 100,
            'requires_fee': False,
            'has_terms': True
        },
        
        # Product announcements with USSD
        {
            'text': f"New! M-PESA Loan Limit increased to Ksh{random.choice([50000, 70000, 100000])}. Borrow at competitive rates. Dial *334# to check your limit. Safaricom - Transforming Lives.",
            'category': 'product_launch',
            'has_prize': False,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': "Introducing M-PESA Global - Send money to over 200 countries at the best rates. Visit your nearest M-PESA agent or dial *334# to get started.",
            'category': 'product_launch',
            'has_prize': False,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        
        # Airtime bonuses with USSD
        {
            'text': f"Safaricom Bonus! Buy Ksh{random.choice([50, 100, 200])} airtime via M-PESA and get {random.choice([25, 50, 100])}MB FREE data. Valid until {random.choice(['31st Dec', 'month end', 'this weekend'])}. Dial *444# to purchase.",
            'category': 'airtime_bonus',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': True,
            'has_terms': False
        },
        {
            'text': f"Weekend Special: Top up Ksh{random.choice([100, 200, 500])} or more via M-PESA and get {random.choice([50, 100, 200])}MB data FREE. Offer ends Sunday midnight. Dial *444# to buy.",
            'category': 'data_bonus',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': True,
            'has_terms': False
        },
        
        # Loyalty rewards
        {
            'text': f"Thank you for being a loyal M-PESA customer for {random.randint(1,10)} years! Enjoy {random.choice(['free transactions', 'reduced charges', '50% off'])} this month. No action needed, automatically applied.",
            'category': 'loyalty',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': f"Valued Customer: As appreciation, you get {random.choice([10, 15, 20])}% discount on all M-PESA charges for the next 30 days. Keep transacting! #CustomerFirst",
            'category': 'loyalty',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        
        # Events/competitions with SMS codes
        {
            'text': f"Enter the Safaricom {random.choice(['Jazz Festival', 'Marathon', 'Innovation Week'])} competition! Stand a chance to win {random.choice(['VIP tickets', 'entry passes', 'backstage access'])} for 2. SMS {random.choice(['JAZZ', 'RUN', 'INNOVATE'])} to 22444. Entry free. Winners announced {random.choice(['20th Jan', '15th Feb', 'next week'])}.",
            'category': 'competition',
            'has_prize': True,
            'prize_amount': 5000,
            'requires_fee': False,
            'has_terms': True
        },
        
        # Information/maintenance (also legit promotions)
        {
            'text': f"Reminder: M-PESA services will be temporarily unavailable on {random.choice(['25th Dec', '1st Jan', '15th Feb'])} from 2AM to 4AM for system maintenance. We apologize for any inconvenience caused.",
            'category': 'maintenance',
            'has_prize': False,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': f"New M-PESA rates effective 1st {random.choice(['Jan', 'Feb', 'Mar'])}. Transaction charges reduced for amounts below Ksh500. Check www.safaricom.co.ke/mpesa for full tariff guide.",
            'category': 'info_update',
            'has_prize': False,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': True
        },
        
        # Bonga points promotions
        {
            'text': f"Your Bonga Points expire on {random.choice(['31st Dec', '31st Jan', 'month end'])}! Redeem now for airtime, data bundles or shopping vouchers. Dial *126# to redeem. Don't let them go to waste!",
            'category': 'bonga_reminder',
            'has_prize': False,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': f"You have {random.randint(1000, 10000)} Bonga Points! Convert them to M-PESA cash or buy data bundles. Dial *126*5# to check balance and redeem. Enjoy!",
            'category': 'bonga_balance',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        }
    ]
    
    promo = random.choice(promo_templates)
    
    # Occasionally add sender variations (Safaricom sends from multiple IDs)
    sender_ids = ['Safaricom', 'MPESA', 'M-PESA', 'BONGA']
    sender_weights = [0.5, 0.3, 0.15, 0.05]  # Safaricom most common
    
    return {
        'message_id': f'PROMO{msg_id:08d}',
        'sender_id': random.choices(sender_ids, weights=sender_weights)[0],
        'message_text': promo['text'],
        'timestamp': timestamp,
        'is_fraud': 0,
        'fraud_type': None,
        'message_type': 'legitimate_promotion',
        'promo_category': promo['category'],
        'has_prize': promo['has_prize'],
        'prize_amount': promo['prize_amount'],
        'requires_fee': promo['requires_fee'],
        'has_terms': promo['has_terms']
    }
    
    timestamp = datetime.now() - timedelta(days=random.randint(0, 90))
    
    promo_templates = [
        # Partner promotions
        {
            'text': "Lipa na M-PESA at Naivas Supermarket this December and stand a chance to win a trolley shopping worth Ksh20,000. Shop smart, pay smarter!",
            'category': 'partner_promo',
            'has_prize': True,
            'prize_amount': 20000,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': "Get 10% cashback when you pay with M-PESA at Carrefour this weekend. Maximum cashback Ksh500. Visit www.safaricom.co.ke/mpesa for terms and conditions.",
            'category': 'cashback',
            'has_prize': True,
            'prize_amount': 500,
            'requires_fee': False,
            'has_terms': True
        },
        {
            'text': "Pay your electricity bill via M-PESA and get 5% instant cashback! Minimum payment Ksh1000. Offer valid until 31st Dec. Terms apply.",
            'category': 'utility_cashback',
            'has_prize': True,
            'prize_amount': 50,
            'requires_fee': False,
            'has_terms': True
        },
        
        # Product announcements
        {
            'text': "New! M-PESA Loan Limit increased to Ksh70,000. Borrow at competitive rates. Dial *334# to check your limit. Safaricom - Transforming Lives.",
            'category': 'product_launch',
            'has_prize': False,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': "Introducing M-PESA Global - Send money to over 200 countries at the best rates. Visit your nearest M-PESA agent or dial *334# to get started.",
            'category': 'product_launch',
            'has_prize': False,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        
        # Airtime/data bonuses
        {
            'text': "Safaricom Bonus! Buy Ksh100 airtime via M-PESA and get 50MB FREE data. Valid until 31st Dec. Dial *444# to purchase. Enjoy!",
            'category': 'airtime_bonus',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': True,  # But transparent - buy airtime
            'has_terms': False
        },
        {
            'text': "Weekend Special: Top up Ksh200 or more via M-PESA and get 100MB data FREE. Offer ends Sunday midnight. *444# to buy.",
            'category': 'data_bonus',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': True,
            'has_terms': False
        },
        
        # Loyalty rewards
        {
            'text': "Thank you for being a loyal M-PESA customer for 5 years! Enjoy free transactions this month. No action needed, automatically applied.",
            'category': 'loyalty',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': "Valued Customer: As appreciation, you get 10% discount on all M-PESA charges for the next 30 days. Keep transacting!",
            'category': 'loyalty',
            'has_prize': True,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        
        # Events/competitions
        {
            'text': "Enter the Safaricom Jazz Festival competition! Stand a chance to win VIP tickets for 2. SMS JAZZ to 22444. Entry free. Winners announced 20th January.",
            'category': 'competition',
            'has_prize': True,
            'prize_amount': 5000,
            'requires_fee': False,
            'has_terms': True
        },
        {
            'text': "M-PESA Foundation Marathon coming soon! Register now at www.safaricom.co.ke/marathon. Entry fee Ksh1000. Proceeds go to charity.",
            'category': 'event',
            'has_prize': False,
            'prize_amount': 0,
            'requires_fee': True,
            'has_terms': True
        },
        
        # Information/updates
        {
            'text': "Reminder: M-PESA services will be temporarily unavailable on 25th Dec from 2AM to 4AM for system maintenance. We apologize for any inconvenience.",
            'category': 'maintenance',
            'has_prize': False,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': False
        },
        {
            'text': "New M-PESA rates effective 1st Jan. Transaction charges reduced for amounts below Ksh500. Check www.safaricom.co.ke/mpesa for full tariff.",
            'category': 'info_update',
            'has_prize': False,
            'prize_amount': 0,
            'requires_fee': False,
            'has_terms': True
        }
    ]
    
    promo = random.choice(promo_templates)
    
    return {
        'message_id': f'PROMO{msg_id:08d}',
        'sender_id': random.choice(['Safaricom', 'MPESA', 'M-PESA']),
        'message_text': promo['text'],
        'timestamp': timestamp,
        'is_fraud': 0,
        'fraud_type': None,
        'message_type': 'legitimate_promotion',
        'promo_category': promo['category'],
        'has_prize': promo['has_prize'],
        'prize_amount': promo['prize_amount'],
        'requires_fee': promo['requires_fee'],
        'has_terms': promo['has_terms']
    }

# ============================================================================
# FRAUDULENT PROMOTION GENERATORS
# ============================================================================

def generate_fake_lottery(msg_id):
    """You won! scams"""
    
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
    
    prizes = [
        ('Ksh500,000', 500000),
        ('Ksh1,000,000', 1000000),
        ('Ksh250,000', 250000),
        ('a brand new car', 2000000),
        ('iPhone 15 Pro', 150000),
        ('a trip to Dubai', 300000)
    ]
    prize_text, prize_value = random.choice(prizes)
    
    templates = [
        f"CONGRATULATIONS!!! You have WON {prize_text} in the Safaricom M-PESA mega lottery! Claim your prize now at https://bit.ly/safaricom-winner{random.randint(100,999)}. Reference: SAF{random.randint(10000,99999)}",
        
        f"🎉🎉🎉 WINNER ALERT! Your M-PESA number has won {prize_text} in our anniversary promotion! Click here to claim: http://mpesa-winners.com/claim?id={random.randint(1000,9999)}",
        
        f"MPESA LOTTERY WINNER! You won {prize_text}!!! To claim, visit www.safaricom-promo.co.ke/winner and enter code: WIN{random.randint(1000,9999)}. HURRY! Expires in 48 hours!!!",
        
        f"LUCKY WINNER NOTIFICATION: Congratulations! You are selected for {prize_text} prize! Confirm at https://bit.ly/mpesa-prize{random.randint(100,999)}. Act now before it expires!",
    ]
    
    return {
        'message_id': f'PROMO{msg_id:08d}',
        'sender_id': random.choice(['MPESA', 'Safaricom', 'M-PESA', '+254700000000', 'PROMO', 'WINNER']),
        'message_text': random.choice(templates),
        'timestamp': timestamp,
        'is_fraud': 1,
        'fraud_type': 'fake_lottery',
        'message_type': 'fraudulent_promotion',
        'prize_amount': prize_value
    }

def generate_fee_scam(msg_id):
    """Pay fee to claim prize"""
    
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
    
    prize_amount = random.choice([100000, 250000, 500000, 750000, 1000000])
    fee = random.choice([500, 1000, 1500, 2000, 2500])
    phone = f"0{random.randint(700000000,799999999)}"
    
    templates = [
        f"Safaricom Promotion: You won Ksh{prize_amount:,}! Pay Ksh{fee} registration fee to {phone} to process your winnings. Valid today only! Don't miss out!!!",
        
        f"M-PESA WINNER NOTIFICATION: Ksh{prize_amount:,} has been reserved for your number. Send Ksh{fee} verification fee to +254{random.randint(700000000,799999999)} to release funds immediately.",
        
        f"URGENT: Your prize of Ksh{prize_amount:,} from Safaricom lottery will expire in 24 hours! Pay Ksh{fee} processing charge to claim. Paybill: {random.randint(100000,999999)} Account: PRIZE",
        
        f"WINNER ALERT! You've won Ksh{prize_amount:,} in M-PESA anniversary draw! To receive, pay delivery fee Ksh{fee} to {phone}. Fast! Limited time!!!",
    ]
    
    return {
        'message_id': f'PROMO{msg_id:08d}',
        'sender_id': random.choice(['MPESA', 'M-PESA', 'Safaricom', 'WINNR', 'PROMO', 'PRIZE']),
        'message_text': random.choice(templates),
        'timestamp': timestamp,
        'is_fraud': 1,
        'fraud_type': 'fee_scam',
        'message_type': 'fraudulent_promotion',
        'prize_amount': prize_amount,
        'scam_fee': fee
    }

def generate_fake_cashback(msg_id):
    """Too-good-to-be-true cashback"""
    
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
    
    cashback_pct = random.choice([50, 100, 150, 200, 500])
    
    templates = [
        f"MEGA OFFER! Get {cashback_pct}% cashback on ALL M-PESA transactions today! Activate now: Send CASH to 22{random.randint(100,999)}. Limited slots available!!!",
        
        f"FLASH SALE! {cashback_pct}% instant cashback on M-PESA payments! Register now: Call 0{random.randint(700000000,799999999)} IMMEDIATELY! First 100 customers only!!",
        
        f"Safaricom SUPER BONUS: Your account qualifies for {cashback_pct}% cashback on everything! Click to activate: https://bit.ly/activate-bonus{random.randint(100,999)} Hurry before slots fill!!!",
        
        f"LIMITED TIME! Get {cashback_pct}% back on all your M-PESA transactions! Activate: Visit www.mpesa-cashback.com/activate. Ends tonight!!!",
    ]
    
    return {
        'message_id': f'PROMO{msg_id:08d}',
        'sender_id': random.choice(['MPESA', 'Safaricom', 'CASHBACK', 'BONUS', 'OFFER']),
        'message_text': random.choice(templates),
        'timestamp': timestamp,
        'is_fraud': 1,
        'fraud_type': 'fake_cashback',
        'message_type': 'fraudulent_promotion',
        'cashback_percent': cashback_pct
    }

def generate_impersonation_scam(msg_id):
    """Fake CEO/official messages"""
    
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
    
    amount = random.choice([50000, 75000, 100000, 150000])
    
    templates = [
        f"Message from Safaricom CEO: As part of our 25th anniversary, we're giving Ksh{amount:,} to 1000 lucky customers. Verify eligibility: http://safaricom-anniversary.co.ke/verify{random.randint(100,999)}",
        
        f"OFFICIAL SAFARICOM NOTICE: Your M-PESA account has been selected for Ksh{amount:,} government relief grant. Claim at: https://bit.ly/relief-{random.randint(1000,9999)}. Authorized by Board of Directors.",
        
        f"MPESA Management Announcement: Congratulations on your loyalty! You qualify for Ksh{amount:,} reward. Confirm identity here: www.mpesa-rewards-official.com. Valid 48 hours.",
        
        f"SAFARICOM BOARD DECISION: Your number selected for Ksh{amount:,} customer appreciation bonus. Click to accept: https://bit.ly/sf-board{random.randint(100,999)}",
    ]
    
    return {
        'message_id': f'PROMO{msg_id:08d}',
        'sender_id': random.choice(['Safaricom', 'MPESA', 'CEO-MSG', 'OFFICIAL', 'BOARD']),
        'message_text': random.choice(templates),
        'timestamp': timestamp,
        'is_fraud': 1,
        'fraud_type': 'impersonation',
        'message_type': 'fraudulent_promotion',
        'prize_amount': amount
    }

def generate_survey_scam(msg_id):
    """Fake surveys with guaranteed prizes"""
    
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
    
    prize_amount = random.choice([5000, 10000, 25000, 50000])
    
    templates = [
        f"Safaricom Customer Survey: Take our 2-minute survey and WIN Ksh{prize_amount:,} guaranteed! Click: http://survey-safaricom.com/{random.randint(1000,9999)}. ALL participants win!!!",
        
        f"MPESA FEEDBACK ALERT: Complete our 3-question form and get Ksh{prize_amount:,} airtime FREE! 100% guaranteed! Link: https://bit.ly/mpesa-survey{random.randint(100,999)}",
        
        f"YOU'RE SELECTED! Safaricom wants your opinion. Answer 5 questions, receive Ksh{prize_amount:,} instantly to your M-PESA: www.safaricom-voice.co.ke. Everyone wins!",
        
        f"QUICK SURVEY = INSTANT MONEY! Share your M-PESA experience (30 seconds) and get Ksh{prize_amount:,}. Click: https://bit.ly/quick-survey{random.randint(100,999)}. No catch!!!",
    ]
    
    return {
        'message_id': f'PROMO{msg_id:08d}',
        'sender_id': random.choice(['Safaricom', 'MPESA', 'SURVEY', 'FEEDBACK', 'RESEARCH']),
        'message_text': random.choice(templates),
        'timestamp': timestamp,
        'is_fraud': 1,
        'fraud_type': 'survey_scam',
        'message_type': 'fraudulent_promotion',
        'prize_amount': prize_amount
    }

def generate_upgrade_scam(msg_id):
    """Fake account upgrades requiring payment"""
    
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
    
    upgrade_fee = random.choice([1000, 1500, 2000, 2500, 3000])
    limit_increase = random.choice([150000, 200000, 300000, 500000])
    
    templates = [
        f"M-PESA UPGRADE ALERT: Your account selected for PREMIUM status! Unlock Ksh{limit_increase:,} daily limit + zero charges. One-time upgrade fee: Ksh{upgrade_fee}. Pay to 0{random.randint(700000000,799999999)}",
        
        f"SAFARICOM VIP MEMBERSHIP: You qualify for Gold Status! Benefits: 0% fees, Ksh{limit_increase:,} limit, priority support. Activation fee: Ksh{upgrade_fee}. Call 0{random.randint(700000000,799999999)} NOW!",
        
        f"MPESA PRO ACCOUNT OFFER: Upgrade today! Get zero transaction fees + Ksh100,000 instant loan. Registration: Ksh{upgrade_fee}. Paybill: {random.randint(100000,999999)} Account: UPGRADE",
        
        f"LIMITED SLOTS! M-PESA Diamond Membership now available. Ksh{limit_increase:,} daily limit, no charges ever. Enrollment: Ksh{upgrade_fee}. https://bit.ly/mpesa-diamond{random.randint(100,999)}",
    ]
    
    return {
        'message_id': f'PROMO{msg_id:08d}',
        'sender_id': random.choice(['MPESA', 'Safaricom', 'UPGRADE', 'VIP', 'PREMIUM']),
        'message_text': random.choice(templates),
        'timestamp': timestamp,
        'is_fraud': 1,
        'fraud_type': 'upgrade_scam',
        'message_type': 'fraudulent_promotion',
        'scam_fee': upgrade_fee
    }

# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_dataset(n_samples=10000):
    """Generate complete promotional messages dataset"""
    
    print("🚀 Generating M-PESA Promotion Fraud Dataset...")
    print(f"📊 Target samples: {n_samples:,}\n")
    
    messages = []
    
    # 60% legitimate promotions
    n_legitimate = int(n_samples * 0.60)
    
    print(f"✅ Generating {n_legitimate:,} legitimate promotions...")
    for i in range(n_legitimate):
        messages.append(generate_legitimate_promotion(i))
        if (i + 1) % 1000 == 0:
            print(f"   Progress: {i+1:,}/{n_legitimate:,}")
    
    # 40% fraudulent promotions
    n_fraud = n_samples - n_legitimate
    
    fraud_generators = [
        generate_fake_lottery,
        generate_fee_scam,
        generate_fake_cashback,
        generate_impersonation_scam,
        generate_survey_scam,
        generate_upgrade_scam
    ]
    
    print(f"🚨 Generating {n_fraud:,} fraudulent promotions...")
    fraud_per_type = n_fraud // len(fraud_generators)
    
    msg_id = n_legitimate
    for generator in fraud_generators:
        for _ in range(fraud_per_type):
            messages.append(generator(msg_id))
            msg_id += 1
    
    # Fill remaining
    while len(messages) < n_samples:
        generator = random.choice(fraud_generators)
        messages.append(generator(msg_id))
        msg_id += 1
    
    df = pd.DataFrame(messages)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add 2% label noise
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
    if 'fraud_type' in df.columns:
        fraud_types = df[df['is_fraud']==1]['fraud_type'].value_counts()
        print(fraud_types.to_string())
    
    print(f"\n📤 Sender ID Distribution:")
    print(df['sender_id'].value_counts().head(10).to_string())
    
    # Save
    output_path = 'data/raw/mpesa_promotion_messages.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset saved to: {output_path}")
    
    # Show samples
    print(f"\n🔍 Sample Messages:\n")
    print("="*70)
    print("LEGITIMATE PROMOTION:")
    print("="*70)
    legit = df[df['is_fraud']==0].iloc[0]
    print(f"Sender: {legit['sender_id']}")
    print(f"Message: {legit['message_text']}\n")
    
    print("="*70)
    print("FRAUDULENT PROMOTION (Fake Lottery):")
    print("="*70)
    fraud = df[df['fraud_type']=='fake_lottery'].iloc[0] if 'fraud_type' in df.columns and 'fake_lottery' in df['fraud_type'].values else df[df['is_fraud']==1].iloc[0]
    print(f"Sender: {fraud['sender_id']}")
    print(f"Message: {fraud['message_text']}\n")
    
    print("="*70)
    print("✨ Dataset generation complete!")
    print("="*70)

if __name__ == "__main__":
    main()