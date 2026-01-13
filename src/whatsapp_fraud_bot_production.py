"""
M-PESA Fraud Detection WhatsApp Bot - PRODUCTION VERSION
FIXED: Auto-loading models on first request
"""

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os
import sys
import re
from pathlib import Path
import logging

# ============================================================================
# 1. LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import fraud detector class (but don't instantiate yet)
try:
    from src.unified_predictor import UnifiedFraudDetector
except ImportError:
    from unified_predictor import UnifiedFraudDetector

# ============================================================================
# 2. GLOBAL VARIABLES & MODEL LOADING
# ============================================================================
app = Flask(__name__)
detector = None  # Start as None

def get_detector():
    """
    Singleton pattern: Only load the heavy models once.
    If they are already loaded, return them immediately.
    """
    global detector
    if detector is None:
        try:
            logger.info("🔄 Initializing Fraud Detector Models...")
            detector = UnifiedFraudDetector()
            logger.info("✅ Models loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}", exc_info=True)
            return None
    return detector

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================
def extract_sender_id(text):
    patterns = [
        r'(?:From|Sender|FROM|SENDER):\s*([A-Z0-9-]+)',
        r'^([A-Z]{3,})\s*[-:]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return 'UNKNOWN'

def format_whatsapp_response(result):
    risk_emoji = {"🔴 CRITICAL": "🚨", "🟠 HIGH": "⚠️", "🟡 MEDIUM": "⚡", "🟢 LOW-MEDIUM": "👀", "🟢 LOW": "✅"}
    emoji = risk_emoji.get(result.get('risk_level'), "ℹ️")
    
    return f"{emoji} *Risk:* {result.get('risk_level')}\n" \
           f"*Prob:* {result.get('fraud_probability', 0)*100:.0f}%\n\n" \
           f"{result.get('recommendation')}"

# ============================================================================
# 4. WEBHOOK ROUTE (The Fix is Here)
# ============================================================================
@app.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    # FIX: Attempt to get the detector. If it's None, try loading it now.
    model = get_detector()

    # If it is STILL None after trying to load, then we are truly broken.
    if model is None:
        resp = MessagingResponse()
        resp.message("⚠️ Bot is waking up. Please try again in 10 seconds.")
        return str(resp)

    # Get Message
    incoming_msg = request.values.get('Body', '').strip()
    sender_number = request.values.get('From', '')
    
    resp = MessagingResponse()
    msg = resp.message()

    if not incoming_msg:
        msg.body("Please forward a message to analyze.")
        return str(resp)

    try:
        logger.info(f"🔍 Analyzing message from {sender_number}...")
        
        sender_id = extract_sender_id(incoming_msg)
        
        # Use 'model' variable, not the global 'detector'
        result = model.predict(incoming_msg, sender_id)
        
        msg.body(format_whatsapp_response(result))
        logger.info(f"✅ Analysis Complete: {result.get('risk_level')}")

    except Exception as e:
        logger.error(f"❌ Prediction Error: {e}", exc_info=True)
        msg.body("⚠️ Error analyzing message. Please try again.")

    return str(resp)

@app.route('/health')
def health():
    # Check status without triggering a full load
    is_loaded = detector is not None
    return {"status": "online", "models_loaded": is_loaded}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)