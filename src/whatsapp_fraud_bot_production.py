"""
M-PESA Scam Detection WhatsApp Bot - PRODUCTION READY
Handles edge cases and uses user-friendly language
"""
import logging
from datetime import datetime, timedelta

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
# CONFIGURATION
# ============================================================================

SESSION_TIMEOUT_MINUTES = 5  # Clear sessions after 5 minutes
MAX_MESSAGE_LENGTH = 1000    # Reject messages longer than this
MAX_RETRIES = 3              # Max times user can retry sender input

# ============================================================================
# GLOBAL STATE
# ============================================================================

app = Flask(__name__)
detector = None
user_sessions = {}  # Store conversation state per user

def get_detector():
    """Lazy-load the scam detector"""
    global detector
    if detector is None:
        logger.info("📦 Loading scam detection models...")
        detector = UnifiedFraudDetector()
        logger.info("✅ Models loaded successfully!")
    return detector


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def clean_expired_sessions():
    """Remove sessions older than timeout"""
    cutoff_time = datetime.now() - timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    expired = [
        user_id for user_id, session in user_sessions.items()
        if session.get('created_at', datetime.now()) < cutoff_time
    ]
    for user_id in expired:
        user_sessions.pop(user_id, None)
        logger.info(f"🧹 Cleaned expired session for {user_id}")


def get_or_create_session(user_id):
    """Get existing session or create new one"""
    clean_expired_sessions()
    
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'created_at': datetime.now(),
            'retry_count': 0
        }
    
    return user_sessions[user_id]


def clear_session(user_id):
    """Clear user session"""
    user_sessions.pop(user_id, None)


# ============================================================================
# MESSAGE VALIDATION
# ============================================================================

def is_valid_message_length(text):
    """Check if message is reasonable length"""
    return 10 <= len(text) <= MAX_MESSAGE_LENGTH


def is_command(text):
    """Check if message is a command"""
    commands = ['help', 'start', 'hi', 'hello', 'menu', 'about', 'info', 'cancel', 'stop']
    return text.lower().strip() in commands


def looks_like_full_message(text):
    """
    Check if text looks like a complete SMS message (not just a sender name)
    
    A full message typically has:
    - Multiple words (>5)
    - Contains common SMS keywords
    - Has punctuation
    """
    words = text.split()
    
    # Check 1: Length
    if len(words) < 5:
        return False
    
    # Check 2: Contains SMS keywords
    sms_keywords = [
        'confirmed', 'received', 'paid', 'ksh', 'balance',
        'transaction', 'mpesa', 'safaricom', 'congratulations',
        'winner', 'prize', 'urgent', 'verify', 'account'
    ]
    
    text_lower = text.lower()
    keyword_found = any(keyword in text_lower for keyword in sms_keywords)
    
    # Check 3: Has sentence structure (contains . or multiple sentences)
    has_punctuation = any(p in text for p in ['.', '!', ',', '?'])
    
    return keyword_found and (len(words) >= 8 or has_punctuation)


def is_valid_sender_format(sender_text):
    """
    Check if user's reply looks like a sender ID
    """
    sender_text = sender_text.strip()
    
    # Must be short
    if len(sender_text) > 20 or len(sender_text) < 2:
        return False
    
    # No spaces allowed (unless it's "M-PESA")
    if ' ' in sender_text and sender_text not in ["M-PESA", "M PESA"]:
        return False
    
    # Common non-sender words
    invalid_words = {
        'ok', 'yes', 'no', 'thanks', 'help', 'about', 'please', 
        'send', 'message', 'from', 'sender', 'the', 'is', 'was'
    }
    if sender_text.lower() in invalid_words:
        return False
    
    # Must contain letters or numbers
    if not any(c.isalnum() for c in sender_text):
        return False
    
    return True


# ============================================================================
# AUTO-EXTRACTION
# ============================================================================

def extract_sender_from_message(message_text):
    """
    Try to auto-extract sender from the message itself
    Returns (sender_id, cleaned_message) or (None, original_message)
    """
    
    # Pattern 1: "MPESA: Confirmed..."
    match = re.match(r'^([A-Z0-9-]+):\s*(.+)', message_text, re.IGNORECASE)
    if match:
        sender = match.group(1).strip()
        message = match.group(2).strip()
        if len(sender) <= 20 and len(message) > 10:
            return sender, message
    
    # Pattern 2: "From SAFARICOM - Confirmed..."
    match = re.match(r'^(?:From|Sender)\s+([A-Z0-9-]+)\s*[-:]\s*(.+)', message_text, re.IGNORECASE)
    if match:
        sender = match.group(1).strip()
        message = match.group(2).strip()
        if len(sender) <= 20 and len(message) > 10:
            return sender, message
    
    # Pattern 3: First line is sender, rest is message
    lines = message_text.strip().split('\n', 1)
    if len(lines) == 2:
        potential_sender = lines[0].strip()
        potential_message = lines[1].strip()
        
        if (len(potential_sender) <= 20 and 
            len(potential_message) > 10 and
            is_valid_sender_format(potential_sender)):
            return potential_sender, potential_message
    
    return None, message_text


# ============================================================================
# RESPONSE FORMATTING
# ============================================================================

def format_analysis_response(result):
    """Format scam detection result for WhatsApp (user-friendly language)"""
    
    # Map technical terms to user-friendly ones
    risk_mapping = {
        "🔴 CRITICAL": ("🚨", "DEFINITELY A SCAM"),
        "🟠 HIGH": ("⚠️", "LIKELY A SCAM"),
        "🟡 MEDIUM": ("⚡", "SUSPICIOUS"),
        "🟢 LOW-MEDIUM": ("👀", "SLIGHTLY SUSPICIOUS"),
        "🟢 LOW": ("✅", "LOOKS SAFE")
    }
    
    emoji, user_friendly_risk = risk_mapping.get(
        result['risk_level'], 
        ("ℹ️", "UNKNOWN")
    )
    
    # Build response
    scam_prob = result['fraud_probability'] * 100
    
    response = f"{emoji} *SCAM ANALYSIS*\n\n"
    response += f"*Assessment:* {user_friendly_risk}\n"
    response += f"*Confidence:* {scam_prob:.0f}%\n\n"
    
    # Recommendation based on probability
    if scam_prob >= 80:
        response += "🛑 *RECOMMENDATION: DELETE THIS MESSAGE*\n\n"
        response += "This is almost certainly a scam. Do NOT:\n"
        response += "❌ Click any links\n"
        response += "❌ Call the numbers\n"
        response += "❌ Share your PIN\n"
        response += "❌ Send money\n\n"
    elif scam_prob >= 50:
        response += "⚠️ *RECOMMENDATION: BE VERY CAREFUL*\n\n"
        response += "This message shows multiple red flags. Verify before taking any action.\n\n"
    elif scam_prob >= 30:
        response += "👀 *RECOMMENDATION: VERIFY FIRST*\n\n"
        response += "Some suspicious elements detected. Double-check before acting.\n\n"
    else:
        response += "✅ *RECOMMENDATION: APPEARS LEGITIMATE*\n\n"
        response += "This message looks safe, but always stay alert.\n\n"
    
    # Add verification tip for risky messages
    if scam_prob >= 40:
        response += "📞 *Verify with Safaricom:*\n"
        response += "• Call 100 (Customer Care)\n"
        response += "• Call 234 (M-PESA)\n"
        response += "• Never share your PIN!\n\n"
    
    # Footer
    response += "_🛡️ Stay safe from scams_\n"
    response += "_Type HELP for more options_"
    
    return response


def format_help_message():
    """Help message"""
    return """👋 *M-PESA Scam Detector*

🔍 *How It Works:*
1. Forward the suspicious SMS to me
2. I'll ask who sent it
3. Get instant scam analysis

💡 *Quick Example:*
```
You: Confirmed. Ksh5000 paid...
Bot: Who sent this?
You: MPESA
Bot: ✅ Analysis complete!
```

📱 *Commands:*
• HELP - Show this menu
• ABOUT - Learn more
• CANCEL - Start over

⚡ *Pro Tips:*
• You can paste "SENDER: message"
• Works with ANY M-PESA/bank SMS
• Results in seconds

_🇰🇪 Protecting Kenya from SMS scams_"""


def format_about_message():
    """About message"""
    return """ℹ️ *About This Bot*

🤖 AI-powered scam detection
🎯 95%+ accuracy
⚡ Real-time analysis
🔒 Your privacy protected

🛡️ *What I Check:*
• Sender authenticity
• Message patterns
• Suspicious links
• Social engineering tactics
• Known scam signatures

⚠️ *Important:*
This tool helps identify scams but isn't perfect. When in doubt:

📞 *Official Contacts:*
• 100 (Safaricom Prepay)
• 200 (Safaricom Postpay)
• 234 (M-PESA Support)

🌐 *Official Website:*
www.safaricom.co.ke

🔐 *Remember:*
• NEVER share your M-PESA PIN
• Safaricom NEVER asks for PINs
• Verify before clicking links

_Built with ❤️ for Kenya 🇰🇪_"""


def format_waiting_for_sender_prompt():
    """Prompt when waiting for sender"""
    return (
        "📥 *Message received!*\n\n"
        "Now I need to know: **Who sent this message?**\n\n"
        "Reply with just the sender name/number as shown in your SMS.\n\n"
        "✅ *Good Examples:*\n"
        "• MPESA\n"
        "• SAFARICOM\n"
        "• EQUITY\n"
        "• 0722000000\n"
        "• M-PESA\n\n"
        "❌ *Don't send:*\n"
        "• \"It came from MPESA\" (too many words)\n"
        "• Another full message\n\n"
        "_Just type the sender and hit send_ ➤"
    )


def format_invalid_sender_error(retry_count):
    """Error message for invalid sender with helpful hints"""
    
    base_msg = (
        "⚠️ *That doesn't look like a sender name*\n\n"
        "I need just the sender ID (the name/number at the top of the SMS).\n\n"
    )
    
    if retry_count == 1:
        base_msg += (
            "💡 *Quick tip:*\n"
            "Don't describe it - just copy the exact sender name!\n\n"
            "✅ Copy this: **MPESA**\n"
            "❌ Not this: \"the message is from mpesa\"\n\n"
        )
    elif retry_count >= 2:
        base_msg += (
            "🤔 *Still having trouble?*\n\n"
            "**Option 1:** Type CANCEL to start fresh\n"
            "**Option 2:** Send your message as:\n"
            "```\n"
            "SENDER_NAME: message text here\n"
            "```\n\n"
            "I can auto-detect the sender that way!\n\n"
        )
    
    base_msg += (
        "📋 *Examples of sender names:*\n"
        "• MPESA\n"
        "• SAFARICOM  \n"
        "• EQUITY\n"
        "• KCB\n"
        "• 0722000000\n\n"
        "_Try again - just the sender name_ ➤"
    )
    
    return base_msg


# ============================================================================
# MAIN WEBHOOK HANDLER
# ============================================================================

@app.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """Main WhatsApp webhook handler with edge case handling"""
    
    # Get detector (loads models if needed)
    model = get_detector()
    
    # Create response
    resp = MessagingResponse()
    msg = resp.message()
    
    # Check if models loaded
    if model is None:
        msg.body("⚠️ Bot is starting up. Please wait 10 seconds and try again! 🚀")
        return str(resp)
    
    # Get user info
    user_id = request.values.get('From', '')
    incoming_msg = request.values.get('Body', '').strip()
    
    logger.info(f"📱 Message from {user_id}: {incoming_msg[:50]}...")
    
    # ========== EMPTY MESSAGE ==========
    if not incoming_msg:
        msg.body(
            "👋 Hi! Send me any suspicious M-PESA or bank message to check if it's a scam.\n\n"
            "Type HELP to learn how to use me!"
        )
        return str(resp)
    
    # ========== COMMANDS ==========
    if is_command(incoming_msg):
        cmd = incoming_msg.lower().strip()
        
        if cmd in ['help', 'start', 'hi', 'hello', 'menu']:
            clear_session(user_id)  # Clear any pending session
            msg.body(format_help_message())
            return str(resp)
        
        if cmd in ['about', 'info']:
            msg.body(format_about_message())
            return str(resp)
        
        if cmd in ['cancel', 'stop', 'reset']:
            clear_session(user_id)
            msg.body(
                "✅ *Session cleared!*\n\n"
                "Send me a new message to analyze, or type HELP for instructions."
            )
            return str(resp)
    
    # ========== MESSAGE LENGTH CHECK ==========
    if not is_valid_message_length(incoming_msg):
        if len(incoming_msg) < 10:
            msg.body(
                "⚠️ *Message too short*\n\n"
                "Please send the complete SMS message you want me to analyze.\n\n"
                "Type HELP if you need instructions!"
            )
        else:
            msg.body(
                "⚠️ *Message too long*\n\n"
                "Please send one message at a time (max 1000 characters).\n\n"
                "If you have multiple messages, send them separately!"
            )
        return str(resp)
    
    # ========== SESSION MANAGEMENT ==========
    session = get_or_create_session(user_id)
    
    # ========== CASE 1: WAITING FOR SENDER NAME ==========
    if session.get('waiting_for_sender'):
        
        # EDGE CASE: User sent another full message instead of sender
        if looks_like_full_message(incoming_msg):
            logger.info(f"⚠️ User {user_id} sent full message when expecting sender")
            
            # Try to auto-extract from new message
            auto_sender, cleaned_msg = extract_sender_from_message(incoming_msg)
            
            if auto_sender:
                # Success! Use this new message
                logger.info(f"✅ Auto-extracted sender from replacement message: {auto_sender}")
                
                clear_session(user_id)
                
                try:
                    result = model.predict(cleaned_msg, auto_sender)
                    msg.body(format_analysis_response(result))
                    logger.info(f"✅ Analysis complete: {result['risk_level']}")
                except Exception as e:
                    logger.error(f"❌ Prediction error: {e}", exc_info=True)
                    msg.body(
                        "⚠️ *Analysis error*\n\n"
                        "Something went wrong. Please try again or type HELP."
                    )
                
                return str(resp)
            
            else:
                # Could not extract, save new message and ask again
                session['message_text'] = incoming_msg
                session['retry_count'] = 0
                
                msg.body(
                    "📝 *New message detected!*\n\n"
                    "I see you sent a different message. No problem!\n\n"
                    + format_waiting_for_sender_prompt()
                )
                return str(resp)
        
        # Validate sender format
        if not is_valid_sender_format(incoming_msg):
            session['retry_count'] = session.get('retry_count', 0) + 1
            
            # Too many retries
            if session['retry_count'] >= MAX_RETRIES:
                logger.info(f"⚠️ User {user_id} exceeded retry limit")
                clear_session(user_id)
                msg.body(
                    "😕 *Having trouble?*\n\n"
                    "Let's start fresh! Send me the full message in this format:\n\n"
                    "```\n"
                    "SENDER_NAME: Your message here...\n"
                    "```\n\n"
                    "Example:\n"
                    "```\n"
                    "MPESA: Confirmed. Ksh500...\n"
                    "```\n\n"
                    "Or type HELP for more options."
                )
                return str(resp)
            
            msg.body(format_invalid_sender_error(session['retry_count']))
            return str(resp)
        
        # VALID SENDER RECEIVED
        message_text = session['message_text']
        sender_id = incoming_msg.strip().upper()
        
        logger.info(f"🔍 Analyzing: sender={sender_id}, msg_length={len(message_text)}")
        
        # Clear session
        clear_session(user_id)
        
        # Run scam detection
        try:
            result = model.predict(message_text, sender_id)
            msg.body(format_analysis_response(result))
            logger.info(f"✅ Analysis complete: {result['risk_level']}")
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            msg.body(
                "⚠️ *Analysis error*\n\n"
                "Something went wrong analyzing this message.\n\n"
                "Please try again or type HELP."
            )
        
        return str(resp)
    
    # ========== CASE 2: NEW MESSAGE TO ANALYZE ==========
    else:
        
        # Try to auto-extract sender
        auto_sender, cleaned_message = extract_sender_from_message(incoming_msg)
        
        if auto_sender:
            # SUCCESS - Found sender in message!
            logger.info(f"✅ Auto-extracted sender: {auto_sender}")
            
            try:
                result = model.predict(cleaned_message, auto_sender)
                msg.body(format_analysis_response(result))
                logger.info(f"✅ Analysis complete: {result['risk_level']}")
            except Exception as e:
                logger.error(f"❌ Prediction error: {e}", exc_info=True)
                msg.body(
                    "⚠️ *Analysis error*\n\n"
                    "Something went wrong. Please try again."
                )
            
            return str(resp)
        
        else:
            # Need to ask for sender
            session['message_text'] = incoming_msg
            session['waiting_for_sender'] = True
            session['retry_count'] = 0
            
            msg.body(format_waiting_for_sender_prompt())
            
            return str(resp)


# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model = get_detector()
    return {
        'status': 'healthy',
        'models_loaded': model is not None,
        'active_sessions': len(user_sessions)
    }, 200


@app.route('/', methods=['GET'])
def home():
    """Home page"""
    model = get_detector()
    status = "✅ ONLINE" if model else "⏳ LOADING..."
    
    return f"""
    <html>
    <head>
        <title>M-PESA Scam Detector Bot</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial; 
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px;
                line-height: 1.6;
            }}
            .status {{ color: green; font-weight: bold; font-size: 24px; }}
            .section {{ 
                background: #f5f5f5; 
                padding: 15px; 
                margin: 20px 0; 
                border-radius: 8px; 
            }}
            code {{ 
                background: #e0e0e0; 
                padding: 2px 6px; 
                border-radius: 3px; 
                font-family: monospace;
            }}
            h3 {{ color: #333; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <h1>🛡️ M-PESA Scam Detector Bot</h1>
        <h2 class="status">{status}</h2>
        
        <div class="section">
            <h3>📊 System Status</h3>
            <ul>
                <li><strong>Models:</strong> {"✅ Loaded" if model else "⏳ Loading..."}</li>
                <li><strong>Active Sessions:</strong> {len(user_sessions)}</li>
                <li><strong>Session Timeout:</strong> {SESSION_TIMEOUT_MINUTES} minutes</li>
            </ul>
        </div>
        
        <div class="section">
            <h3>📱 How to Use</h3>
            <ol>
                <li>Add the WhatsApp bot number to your contacts</li>
                <li>Send any suspicious M-PESA or bank SMS</li>
                <li>Reply with the sender name when prompted</li>
                <li>Get instant scam analysis</li>
            </ol>
            
            <p><strong>💡 Pro Tip:</strong> You can send messages in this format for instant analysis:</p>
            <code>SENDER_NAME: Message text here...</code>
        </div>
        
        <div class="section">
            <h3>🔗 API Endpoints</h3>
            <ul>
                <li><code>/whatsapp</code> - Main webhook (POST)</li>
                <li><code>/health</code> - Health check (GET)</li>
                <li><code>/</code> - This page (GET)</li>
            </ul>
        </div>
        
        <div class="section">
            <h3>🛠️ Features</h3>
            <ul>
                <li>✅ Auto-detects sender from formatted messages</li>
                <li>✅ Handles edge cases (multiple retries, wrong input)</li>
                <li>✅ Session management with automatic timeout</li>
                <li>✅ User-friendly language ("scam" instead of "fraud")</li>
                <li>✅ Clear error messages and helpful hints</li>
                <li>✅ Commands: HELP, ABOUT, CANCEL</li>
            </ul>
        </div>
        
        <hr>
        <p><small>🇰🇪 M-PESA Scam Detector v2.0 | Built with Flask + Twilio + AI</small></p>
    </body>
    </html>
    """


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("="*70)
    print("🚀 M-PESA SCAM DETECTOR WHATSAPP BOT v2.0")
    print("="*70)
    print(f"\n📱 Webhook: http://localhost:{port}/whatsapp")
    print(f"🏥 Health: http://localhost:{port}/health")
    print(f"🌐 Home: http://localhost:{port}/")
    print(f"\n⚙️  Settings:")
    print(f"   • Session timeout: {SESSION_TIMEOUT_MINUTES} minutes")
    print(f"   • Max message length: {MAX_MESSAGE_LENGTH} chars")
    print(f"   • Max retries: {MAX_RETRIES}")
    print("\n" + "="*70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)