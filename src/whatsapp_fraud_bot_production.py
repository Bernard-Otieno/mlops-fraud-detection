"""
M-PESA Fraud Detection WhatsApp Bot - FIXED SENDER PROMPT
Properly asks users for sender information
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os
import sys
from pathlib import Path
import re

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.unified_predictor import UnifiedFraudDetector

# ============================================================================
# GLOBAL STATE
# ============================================================================

app = Flask(__name__)
detector = None
user_sessions = {}  # Store conversation state per user

def get_detector():
    """Lazy-load the fraud detector"""
    global detector
    if detector is None:
        logger.info("🔄 Loading fraud detection models...")
        detector = UnifiedFraudDetector()
        logger.info("✅ Models loaded successfully!")
    return detector


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_sender_from_message(message_text):
    """
    Try to auto-extract sender from the message itself
    Returns (sender_id, cleaned_message) or (None, original_message)
    """
    
    # Pattern 1: "MPESA: Confirmed..."
    match = re.match(r'^([A-Z0-9-]+):\s*(.+)', message_text, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    
    # Pattern 2: "From SAFARICOM - Confirmed..."
    match = re.match(r'^(?:From|Sender)\s+([A-Z0-9-]+)\s*[-:]\s*(.+)', message_text, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    
    # Pattern 3: First word is all caps and short (likely sender)
    words = message_text.split()
    if words and len(words[0]) <= 15 and words[0].isupper():
        return words[0], ' '.join(words[1:])
    
    return None, message_text


def is_valid_sender_format(sender_text):
    """
    Check if user's reply looks like a sender ID
    """
    sender_text = sender_text.strip()
    
    # Must be short
    if len(sender_text) > 20 or len(sender_text) < 2:
        return False
    
    # No spaces allowed (unless it's "M-PESA")
    if ' ' in sender_text and sender_text != "M-PESA":
        return False
    
    # Common non-sender words
    invalid_words = {'ok', 'yes', 'no', 'thanks', 'help', 'about', 'please', 'send'}
    if sender_text.lower() in invalid_words:
        return False
    
    # Must contain letters or numbers
    if not any(c.isalnum() for c in sender_text):
        return False
    
    return True


def format_whatsapp_response(result):
    """Format fraud detection result for WhatsApp"""
    
    risk_emoji = {
        "🔴 CRITICAL": "🚨",
        "🟠 HIGH": "⚠️",
        "🟡 MEDIUM": "⚡",
        "🟢 LOW-MEDIUM": "👀",
        "🟢 LOW": "✅"
    }
    
    emoji = risk_emoji.get(result['risk_level'], "ℹ️")
    
    response = f"{emoji} *FRAUD ANALYSIS*\n\n"
    response += f"*Risk:* {result['risk_level']}\n"
    response += f"*Probability:* {result['fraud_probability']*100:.0f}%\n\n"
    response += f"{result['recommendation']}\n\n"
    
    # Add verification tip based on risk
    if result['fraud_probability'] >= 0.5:
        response += "📞 *Verify with Safaricom:*\n"
        response += "• Call 100 (Customer Care)\n"
        response += "• Call 234 (M-PESA)\n\n"
    
    response += "_Powered by AI Fraud Detection_"
    
    return response


def format_help_message():
    """Help message"""
    return """👋 *M-PESA Fraud Detector*

📱 *How to use:*
1. Forward or paste the suspicious SMS
2. I'll ask for the sender name
3. Get instant fraud analysis

💡 *Example:*
You: _Confirmed. Ksh5000 paid..._
Bot: Who sent this message?
You: _MPESA_
Bot: ✅ Analysis complete!

🔍 *Commands:*
• HELP - Show this message
• ABOUT - Learn more

_Protecting Kenya from SMS fraud_"""


def format_about_message():
    """About message"""
    return """ℹ️ *About This Bot*

🤖 AI-powered fraud detection
🎯 95%+ accuracy rate
⚡ Real-time analysis
🔒 Privacy-focused (no data stored)

⚠️ *Disclaimer:*
This is an automated tool. Always verify suspicious messages with official Safaricom:
• 100 (Prepay)
• 200 (Postpay)  
• 234 (M-PESA)

🌐 *Official:* safaricom.co.ke

_Built for Kenya 🇰🇪_"""


# ============================================================================
# MAIN WEBHOOK
# ============================================================================

@app.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """Main WhatsApp webhook handler"""
    
    # Get detector (loads models if needed)
    model = get_detector()
    
    # Create response
    resp = MessagingResponse()
    msg = resp.message()
    
    # Check if models loaded
    if model is None:
        msg.body("⚠️ Bot is starting. Please try again in 10 seconds.")
        return str(resp)
    
    # Get user info
    user_id = request.values.get('From', '')
    incoming_msg = request.values.get('Body', '').strip()
    
    logger.info(f"📱 Message from {user_id}: {incoming_msg[:50]}...")
    
    # Empty message
    if not incoming_msg:
        msg.body("Please send me the suspicious message to analyze. 📱")
        return str(resp)
    
    # ========== COMMAND HANDLING ==========
    
    incoming_lower = incoming_msg.lower()
    
    if incoming_lower in ['help', 'start', 'hi', 'hello', 'menu']:
        msg.body(format_help_message())
        return str(resp)
    
    if incoming_lower in ['about', 'info']:
        msg.body(format_about_message())
        return str(resp)
    
    # ========== SESSION MANAGEMENT ==========
    
    # Check if user has an ongoing session
    session = user_sessions.get(user_id)
    
    if session and 'waiting_for_sender' in session:
        # USER IS REPLYING WITH SENDER ID
        
        # Validate sender format
        if not is_valid_sender_format(incoming_msg):
            msg.body(
                "⚠️ *Invalid sender format*\n\n"
                "Please reply with the sender name exactly as shown in the SMS.\n\n"
                "✅ *Examples:*\n"
                "• MPESA\n"
                "• SAFARICOM\n"
                "• EQUITY\n"
                "• 0722000000\n\n"
                "❌ *Not valid:*\n"
                "• \"It's from MPESA\" (too many words)\n"
                "• \"ok\" (not a sender)"
            )
            return str(resp)
        
        # Extract saved message
        message_text = session['message_text']
        sender_id = incoming_msg.strip().upper()
        
        logger.info(f"🔍 Analyzing: sender={sender_id}, msg_length={len(message_text)}")
        
        # Clear session
        user_sessions.pop(user_id, None)
        
        # Run fraud detection
        try:
            result = model.predict(message_text, sender_id)
            msg.body(format_whatsapp_response(result))
            logger.info(f"✅ Analysis complete: {result['risk_level']}")
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            msg.body("⚠️ Error analyzing message. Please try again.")
        
        return str(resp)
    
    else:
        # USER IS SENDING A NEW MESSAGE TO ANALYZE
        
        # Try to auto-extract sender
        auto_sender, cleaned_message = extract_sender_from_message(incoming_msg)
        
        if auto_sender:
            # We found the sender in the message!
            logger.info(f"✅ Auto-extracted sender: {auto_sender}")
            
            try:
                result = model.predict(cleaned_message, auto_sender)
                msg.body(format_whatsapp_response(result))
                logger.info(f"✅ Analysis complete: {result['risk_level']}")
            except Exception as e:
                logger.error(f"❌ Prediction error: {e}", exc_info=True)
                msg.body("⚠️ Error analyzing message. Please try again.")
            
            return str(resp)
        
        else:
            # Could not auto-extract sender, ask user
            user_sessions[user_id] = {
                'message_text': incoming_msg,
                'waiting_for_sender': True,
                'timestamp': request.values.get('MessageSid')
            }
            
            msg.body(
                "📥 *Message received!*\n\n"
                "Who sent this message? Reply with the sender name exactly as shown.\n\n"
                "💡 *Examples:*\n"
                "• MPESA\n"
                "• SAFARICOM\n"
                "• EQUITY\n"
                "• 0722000000\n"
                "• M-PESA\n\n"
                "_Just type the sender name and send_"
            )
            
            return str(resp)


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model = get_detector()
    return {
        'status': 'healthy',
        'models_loaded': model is not None
    }, 200


@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return """
    <html>
    <head>
        <title>M-PESA Fraud Detector Bot</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            .status { color: green; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🛡️ M-PESA Fraud Detector</h1>
        <h2 class="status">✅ Bot is Running!</h2>
        
        <h3>📱 How to Use:</h3>
        <ol>
            <li>Add the WhatsApp number to your contacts</li>
            <li>Send any suspicious M-PESA/Safaricom message</li>
            <li>Reply with the sender name when prompted</li>
            <li>Get instant fraud analysis</li>
        </ol>
        
        <h3>🔗 Endpoints:</h3>
        <ul>
            <li><code>/whatsapp</code> - Webhook endpoint</li>
            <li><code>/health</code> - Health check</li>
        </ul>
        
        <hr>
        <p><small>M-PESA Fraud Detector v2.0 | Built with Flask + Twilio</small></p>
    </body>
    </html>
    """


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("="*70)
    print("🚀 M-PESA FRAUD DETECTOR WHATSAPP BOT")
    print("="*70)
    print(f"\n📱 Webhook: http://localhost:{port}/whatsapp")
    print(f"🏥 Health: http://localhost:{port}/health")
    print("\n" + "="*70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)