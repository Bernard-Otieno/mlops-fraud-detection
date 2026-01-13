"""
M-PESA Fraud Detection WhatsApp Bot - PRODUCTION VERSION
FIXED: Windows Unicode Compatibility & Robust Sender Extraction
"""

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os
import sys
import io
import re
from pathlib import Path
import logging
from datetime import datetime

# ============================================================================
# FIX 1: WINDOWS UNICODE & EMOJI SUPPORT
# ============================================================================
if sys.platform == 'win32':
    # Force the console output to use UTF-8 to prevent 'charmap' errors
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Setup logging with explicit UTF-8 encoding for the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot_usage.log', encoding='utf-8')  # CRITICAL FIX
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import fraud detector
try:
    from src.unified_predictor import UnifiedFraudDetector
    logger.info("✅ Successfully imported UnifiedFraudDetector")
except ImportError:
    from unified_predictor import UnifiedFraudDetector

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024

# Initialize fraud detector
logger.info("🔄 Loading fraud detection models...")
try:
    detector = UnifiedFraudDetector()
    logger.info("✅ Models loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    detector = None

usage_stats = {
    'total_messages': 0,
    'fraud_detected': 0,
    'legitimate': 0,
    'errors': 0,
    'unique_users': set()
}

# ============================================================================
# FIX 2: IMPROVED ACCURACY (REGEX SENDER EXTRACTION)
# ============================================================================

def extract_sender_id(text):
    """
    Improved extraction logic using Regex (from your other bot script).
    Better at identifying scammers posing as MPESA or using random numbers.
    """
    patterns = [
        r'(?:From|Sender|FROM|SENDER):\s*([A-Z0-9-]+)',
        r'^([A-Z]{3,})\s*[-:]',  # Matches "MPESA:" or "Safaricom-"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return 'UNKNOWN'

def log_usage(sender, message_length, prediction):
    usage_stats['total_messages'] += 1
    usage_stats['unique_users'].add(sender)
    if prediction == 'fraud':
        usage_stats['fraud_detected'] += 1
    elif prediction == 'legitimate':
        usage_stats['legitimate'] += 1
    logger.info(f"USER: {sender[-10:]} | MSG_LEN: {message_length} | PREDICTION: {prediction}")

def format_whatsapp_response(result):
    risk_emoji = {"🔴 CRITICAL": "🚨", "🟠 HIGH": "⚠️", "🟡 MEDIUM": "⚡", "🟢 LOW-MEDIUM": "👀", "🟢 LOW": "✅"}
    emoji = risk_emoji.get(result['risk_level'], "ℹ️")
    
    response = f"{emoji} *FRAUD ANALYSIS*\n\n"
    response += f"*Risk:* {result['risk_level']}\n"
    response += f"*Fraud Probability:* {result['fraud_probability']*100:.0f}%\n\n"
    response += f"*Recommendation:*\n{result['recommendation']}\n\n"
    
    if result['fraud_indicators']:
        response += "*🔍 Key Findings:*\n"
        for indicator in result['fraud_indicators'][:3]:
            response += f"• {indicator}\n"
    
    response += f"\n_M-PESA Fraud Detector v1.0_"
    return response

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    if detector is None:
        resp = MessagingResponse()
        resp.message("⚠️ Bot is currently offline for maintenance.")
        return str(resp)
    
    incoming_msg = request.values.get('Body', '').strip()
    sender_number = request.values.get('From', '')
    
    logger.info(f"📱 Processing message from {sender_number}")
    
    resp = MessagingResponse()
    msg = resp.message()
    
    if not incoming_msg:
        msg.body("Please send a message to analyze. 📱")
        return str(resp)
    
    incoming_lower = incoming_msg.lower().strip()
    
    # Handle basic commands
    if incoming_lower in ['help', 'start', 'hi', 'hello']:
        msg.body("Forward any suspicious M-PESA message here to analyze it.")
        return str(resp)

    # Analyze message for fraud
    try:
        logger.info("🔍 Analyzing message...")
        
        # USE THE IMPROVED EXTRACTION
        sender_id = extract_sender_id(incoming_msg)
        
        result = detector.predict(incoming_msg, sender_id)
        logger.info(f"✅ Analysis complete: {result['risk_level']}")
        
        prediction_type = 'fraud' if result['is_fraud'] else 'legitimate'
        log_usage(sender_number, len(incoming_msg), prediction_type)
        
        msg.body(format_whatsapp_response(result))
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        usage_stats['errors'] += 1
        msg.body("⚠️ Sorry, I encountered an error. Please try again.")
    
    return str(resp)

@app.route('/health', methods=['GET'])
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}

@app.route('/', methods=['GET'])
def home():
    return "<h1>🛡️ M-PESA Fraud Detector is Online</h1>"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)