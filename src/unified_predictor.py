"""
M-PESA Fraud Detection Bot - RAILWAY STABLE VERSION
"""
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os
import sys
import logging
import re
from pathlib import Path

# ============================================================================
# 1. LOGGING SETUP (Railway Friendly)
# ============================================================================
# We use StreamHandler (Console) only. FileHandler causes crashes on read-only systems.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 2. MODEL LOADER
# ============================================================================
# Ensure we can find the src folder
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

detector = None

def load_models():
    """Safely load models without crashing the server"""
    global detector
    try:
        logger.info("🔄 Attempting to load UnifiedFraudDetector...")
        # Try importing from src first
        try:
            from src.unified_predictor import UnifiedFraudDetector
        except ImportError:
            from unified_predictor import UnifiedFraudDetector
            
        detector = UnifiedFraudDetector()
        logger.info("✅ Models loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Model Loading Failed: {e}", exc_info=True)
        detector = None

# Load models immediately on startup
load_models()

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================
def extract_sender_id(text):
    """Extract sender from forwarded messages (e.g. 'From: MPESA')"""
    patterns = [
        r'(?:From|Sender|FROM|SENDER):\s*([A-Z0-9-]+)',
        r'^([A-Z]{3,})\s*[-:]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return 'UNKNOWN'

def format_response(result):
    risk_emoji = {"🔴 CRITICAL": "🚨", "🟠 HIGH": "⚠️", "🟡 MEDIUM": "⚡", "🟢 LOW-MEDIUM": "👀", "🟢 LOW": "✅"}
    emoji = risk_emoji.get(result.get('risk_level'), "ℹ️")
    return f"{emoji} *Risk:* {result.get('risk_level')}\n*Prob:* {result.get('fraud_probability', 0)*100:.0f}%\n\n{result.get('recommendation')}"

# ============================================================================
# 4. FLASK APP
# ============================================================================
app = Flask(__name__)

@app.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    # 1. Check if models are loaded
    if detector is None:
        # Try to reload one last time
        load_models()
        if detector is None:
            resp = MessagingResponse()
            resp.message("⚠️ System starting up. Please try again in 1 minute.")
            return str(resp)

    # 2. Get Message
    incoming_msg = request.values.get('Body', '').strip()
    
    resp = MessagingResponse()
    msg = resp.message()

    if not incoming_msg:
        msg.body("Please forward a message to analyze.")
        return str(resp)

    # 3. Analyze
    try:
        sender = extract_sender_id(incoming_msg)
        logger.info(f"🔍 Analyzing message from sender: {sender}")
        
        result = detector.predict(incoming_msg, sender)
        
        msg.body(format_response(result))
        logger.info(f"✅ Result: {result.get('risk_level')}")

    except Exception as e:
        logger.error(f"❌ Prediction Error: {e}", exc_info=True)
        msg.body("⚠️ Error analyzing message. Please try again.")

    return str(resp)

@app.route('/health')
def health():
    status = "healthy" if detector else "degraded"
    return {"status": status, "models_loaded": detector is not None}, 200

# 5. ERROR HANDLER (Catches the 500 errors so you can see them!)
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"💥 Unhandled Server Error: {e}", exc_info=True)
    return "Internal Server Error", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)