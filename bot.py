"""
FinanceBot — Telegram + Google Gemini
Rate limiting: 7 messages/day per user (stored in rate_limits.json)
KB questions (company info + finance FAQ) are ALWAYS answered even after limit.
"""

import os
import json
import threading
from datetime import date
from flask import Flask
from threading import Thread
import telebot
import logging
import ssl
import certifi
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# KEEP ALIVE SERVER
# ──────────────────────────────────────────────
app = Flask('')

@app.route('/')
def home():
    return "FinanceBot is running!"

def run():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.start()

keep_alive()

# ──────────────────────────────────────────────
# SSL FIX (Windows)
# ──────────────────────────────────────────────
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs.setdefault('verify', False)
    return original_request(self, method, url, **kwargs)
requests.Session.request = patched_request

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ──────────────────────────────────────────────
# ENV + LOGGING
# ──────────────────────────────────────────────
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
DAILY_LIMIT        = int(os.getenv("DAILY_MESSAGE_LIMIT", "7"))

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set in .env file")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env file")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# ──────────────────────────────────────────────
# RATE LIMITER  (JSON-backed, per user, daily)
# ──────────────────────────────────────────────
RATE_LIMIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate_limits.json")
_rl_lock = threading.Lock()

LIMIT_REACHED_MSG = (
    "⚠️ *Daily limit reached!*\n\n"
    f"You've used all {DAILY_LIMIT} free messages for today.\n"
    "Your limit resets at midnight 🌙\n\n"
    "Come back tomorrow for more finance tips!\n\n"
    "_TradeVed FinanceBot — free tier_"
)


def _today() -> str:
    return date.today().isoformat()


def _load_rl() -> dict:
    if not os.path.exists(RATE_LIMIT_FILE):
        return {}
    try:
        with open(RATE_LIMIT_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_rl(data: dict) -> None:
    with open(RATE_LIMIT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def is_rate_limited(user_id: int) -> bool:
    """Returns True if this user has hit their daily limit."""
    with _rl_lock:
        data  = _load_rl()
        key   = str(user_id)
        entry = data.get(key)
        if not entry or entry.get("date") != _today():
            return False
        return entry.get("count", 0) >= DAILY_LIMIT


def increment_message_count(user_id: int) -> int:
    """Increment today's count for this user. Returns new count."""
    with _rl_lock:
        data  = _load_rl()
        key   = str(user_id)
        entry = data.get(key, {})
        if entry.get("date") != _today():
            entry = {"date": _today(), "count": 0}
        entry["count"] += 1
        data[key] = entry
        _save_rl(data)
        return entry["count"]


def get_remaining(user_id: int) -> int:
    """How many messages this user has left today."""
    with _rl_lock:
        data  = _load_rl()
        key   = str(user_id)
        entry = data.get(key)
        if not entry or entry.get("date") != _today():
            return DAILY_LIMIT
        return max(0, DAILY_LIMIT - entry.get("count", 0))


# ──────────────────────────────────────────────
# KNOWLEDGE BASE
# ──────────────────────────────────────────────
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")

def load_knowledge_base():
    entries = []
    for filename in ["finance_faq.json", "company_info.json"]:
        filepath = os.path.join(KB_DIR, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.extend(data)
                logger.info(f"Loaded {len(data)} entries from {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
    return entries

KNOWLEDGE_BASE = load_knowledge_base()
logger.info(f"Total knowledge base entries: {len(KNOWLEDGE_BASE)}")


def search_knowledge_base(user_message: str) -> str | None:
    """
    Keyword-scored search across all KB entries.
    Returns best answer string if score >= 1, else None.
    """
    query      = user_message.lower().strip()
    best_match = None
    best_score = 0

    for entry in KNOWLEDGE_BASE:
        score = 0
        for kw in entry.get("keywords", []):
            if kw.lower() in query:
                score += len(kw.lower().split())   # longer keyword = higher weight
        if score > best_score:
            best_score = score
            best_match = entry

    if best_score >= 1 and best_match:
        logger.info(f"KB hit (score={best_score}): {best_match['question']}")
        return best_match["answer"]

    return None


# ──────────────────────────────────────────────
# GEMINI LLM
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are FinanceBot, an AI assistant exclusively for Indian personal finance.

You ONLY answer questions about:
- Stock market (NSE/BSE, Sensex, Nifty)
- Mutual funds, SIPs, ELSS, index funds
- Personal finance (savings, budgeting, emergency fund)
- Insurance (term life, health, ULIP)
- Taxes (ITR filing, Section 80C, 80D, capital gains tax)
- Fixed deposits, PPF, NPS, bonds
- IPOs, F&O basics
- Financial planning for salaried Indians

STRICT RULES:
1. If asked ANYTHING outside finance, reply ONLY:
   I am FinanceBot - your personal finance assistant! I only answer finance questions. Try asking about SIPs, mutual funds, tax saving under 80C, or stock market basics!

2. Respond in the SAME language as the user (Hindi/English/Hinglish).

3. Keep answers under 200 words.

4. Always use Indian context: INR, SEBI, RBI, NSE/BSE.

5. NEVER give direct buy/sell advice. Always add: This is educational only, not financial advice. Consult a SEBI-registered advisor."""


def ask_gemini(user_message: str) -> str:
    try:
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            system_instruction=SYSTEM_PROMPT
        )
        response = model.generate_content(user_message)
        return response.text
    except Exception as e:
        logger.error(f"Gemini error: {type(e).__name__}: {str(e)[:200]}")
        return "⚠️ Sorry, I'm having trouble connecting right now. Please try again in a moment!"


# ──────────────────────────────────────────────
# BOT COMMANDS
# ──────────────────────────────────────────────
@bot.message_handler(commands=["start"])
def start(message):
    name = message.from_user.first_name or "there"
    remaining = get_remaining(message.from_user.id)
    reply = (
        f"Hi {name}! I am FinanceBot - your Indian personal finance assistant.\n\n"
        "I can help with:\n"
        "- Stock market (NSE/BSE)\n"
        "- Mutual funds and SIPs\n"
        "- Tax saving (80C, ITR filing)\n"
        "- FDs, PPF, NPS\n"
        "- Insurance guidance\n"
        "- Budgeting and financial planning\n"
        "- TradeVed platform questions\n\n"
        f"💬 Free messages remaining today: {remaining}/{DAILY_LIMIT}\n\n"
        "Just type your finance question!"
    )
    bot.reply_to(message, reply)


@bot.message_handler(commands=["help"])
def help_command(message):
    reply = (
        "Example questions you can ask me:\n\n"
        "- What is a SIP and how do I start one?\n"
        "- How do I save tax under Section 80C?\n"
        "- Difference between Nifty and Sensex?\n"
        "- FD or mutual funds - which is better?\n"
        "- How does capital gains tax work in India?\n"
        "- Explain term insurance in simple words\n"
        "- What is TradeVed?\n\n"
        "Just type your question!"
    )
    bot.reply_to(message, reply)


@bot.message_handler(commands=["about"])
def about(message):
    reply = (
        "About FinanceBot\n\n"
        "A finance-only AI assistant for Indian retail investors by TradeVed.\n"
        "Powered by Google Gemini AI.\n\n"
        "Always consult a SEBI-registered advisor before investing."
    )
    bot.reply_to(message, reply)


@bot.message_handler(commands=["limit"])
def check_limit(message):
    remaining = get_remaining(message.from_user.id)
    if remaining > 0:
        bot.reply_to(message, f"💬 You have *{remaining}* free message(s) left today.\nLimit resets at midnight 🌙", parse_mode="Markdown")
    else:
        bot.reply_to(message, f"⚠️ You've used all *{DAILY_LIMIT}* messages for today.\nLimit resets at midnight 🌙", parse_mode="Markdown")


# ──────────────────────────────────────────────
# MAIN MESSAGE HANDLER
# ──────────────────────────────────────────────
@bot.message_handler(content_types=["text"])
def handle_message(message):
    user_text = message.text
    user      = message.from_user
    user_id   = user.id

    logger.info(f"Message from {user.first_name} ({user_id}): {user_text[:60]}")
    bot.send_chat_action(message.chat.id, "typing")

    # ── Step 1: Always check KB first (free, no limit applied) ──────────────
    kb_answer = search_knowledge_base(user_text)
    if kb_answer:
        bot.reply_to(message, kb_answer)
        logger.info(f"Replied [{user.first_name}] source=KB (no limit consumed)")
        return

    # ── Step 2: Check rate limit BEFORE calling Gemini ──────────────────────
    if is_rate_limited(user_id):
        logger.info(f"Rate limited: user={user_id} ({user.first_name})")
        bot.reply_to(message, LIMIT_REACHED_MSG, parse_mode="Markdown")
        return

    # ── Step 3: Call Gemini and increment count ──────────────────────────────
    reply = ask_gemini(user_text)
    count = increment_message_count(user_id)
    remaining = DAILY_LIMIT - count

    # Only notify on the very last message
    if remaining == 0:
        reply += f"\n\n_⚠️ This was your last free message for today. Your limit resets at midnight 🌙_"

    bot.reply_to(message, reply, parse_mode="Markdown")
    logger.info(f"Replied [{user.first_name}] source=Gemini | used={count}/{DAILY_LIMIT}")


# ──────────────────────────────────────────────
# NON-TEXT HANDLER
# ──────────────────────────────────────────────
@bot.message_handler(func=lambda m: True, content_types=["photo", "voice", "video", "document"])
def handle_non_text(message):
    bot.reply_to(message, "I can only read text messages! Type your finance question.")


# ──────────────────────────────────────────────
# START
# ──────────────────────────────────────────────
logger.info("Starting FinanceBot...")
logger.info(f"Daily message limit: {DAILY_LIMIT} per user")
logger.info(f"Rate limit file: {RATE_LIMIT_FILE}")

# Clear any registered webhook before polling.
# Prevents Error 409 caused by a leftover webhook or
# a previous Render instance still running during redeploy.
bot.delete_webhook(drop_pending_updates=True)
logger.info("Webhook cleared. Starting polling...")

bot.infinity_polling(none_stop=True, interval=1)
