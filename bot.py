"""
FinanceBot — Telegram + Google Gemini
SSL fix for Windows included
"""

import os

import json

import logging
import ssl
import certifi
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# Fix SSL on Windows BEFORE importing telebot
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

# Patch requests to disable SSL verify as fallback
original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs.setdefault('verify', False)
    return original_request(self, method, url, **kwargs)
requests.Session.request = patched_request

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import telebot

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

# --------------- Knowledge Base Setup ---------------
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")

def load_knowledge_base():
    """Load all JSON knowledge base files from the knowledge_base/ folder."""
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


def search_knowledge_base(user_message):
    """
    Search the knowledge base for a matching answer.
    Uses keyword matching with a scoring system.
    Returns the best answer if score exceeds threshold, else None.
    """
    query = user_message.lower().strip()
    best_match = None
    best_score = 0

    for entry in KNOWLEDGE_BASE:
        keywords = entry.get("keywords", [])
        score = 0
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in query:
                # Longer keyword matches get higher weight
                score += len(kw_lower.split())
        if score > best_score:
            best_score = score
            best_match = entry

    # Require at least a score of 1 for a match
    if best_score >= 1 and best_match:
        logger.info(f"KB match found (score={best_score}): {best_match['question']}")
        return best_match["answer"]

    return None
# ----------------------------------------------------

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


def ask_gemini(user_message):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-pro",

            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT
        )
        response = model.generate_content(user_message)
        return response.text
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Sorry, I ran into an issue. Please try again in a moment."


@bot.message_handler(commands=["start"])
def start(message):
    name = message.from_user.first_name or "there"
    reply = (
        f"Hi {name}! I am FinanceBot - your Indian personal finance assistant.\n\n"
        "I can help with:\n"
        "- Stock market (NSE/BSE)\n"
        "- Mutual funds and SIPs\n"
        "- Tax saving (80C, ITR filing)\n"
        "- FDs, PPF, NPS\n"
        "- Insurance guidance\n"
        "- Budgeting and financial planning\n\n"
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
        "- Explain term insurance in simple words\n\n"
        "Just type your question!"
    )
    bot.reply_to(message, reply)


@bot.message_handler(commands=["about"])
def about(message):
    reply = (
        "About FinanceBot\n\n"
        "A finance-only AI assistant for Indian retail investors by Tradeved.\n"
        "Powered by Google Gemini AI.\n\n"
        "Always consult a SEBI-registered advisor before investing."
    )
    bot.reply_to(message, reply)


@bot.message_handler(content_types=["text"])
def handle_message(message):
    user_text = message.text
    user = message.from_user
    logger.info(f"Message from {user.first_name} ({user.id}): {user_text[:60]}")
    bot.send_chat_action(message.chat.id, "typing")

    # Step 1: Check knowledge base first
    kb_answer = search_knowledge_base(user_text)
    if kb_answer:
        bot.reply_to(message, kb_answer)
        logger.info(f"Replied to {user.first_name} [source: knowledge base]")
        return

    # Step 2: Fall back to Gemini LLM
    reply = ask_gemini(user_text)
    bot.reply_to(message, reply)
    logger.info(f"Replied to {user.first_name} [source: Gemini LLM]")
    reply = ask_gemini(user_text)
    bot.reply_to(message, reply)
    logger.info(f"Replied to {user.first_name}")



@bot.message_handler(func=lambda m: True, content_types=["photo", "voice", "video", "document"])
def handle_non_text(message):
    bot.reply_to(message, "I can only read text messages! Type your finance question.")


logger.info("Starting FinanceBot...")
logger.info("Bot is running! Press Ctrl+C to stop.")
bot.infinity_polling()
