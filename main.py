"""Telegram bot for summarizing chat messages using OpenAI."""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import List

import openai
from dotenv import find_dotenv, load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)

load_dotenv(find_dotenv())

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

client = openai.OpenAI()


def load_settings(file_path: str) -> dict:
    """Load settings from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("Settings file not found: %s", file_path)
        return {}
    except json.JSONDecodeError:
        logging.error("Invalid JSON in settings file: %s", file_path)
        return {}

SETTINGS = load_settings("settings.json")
ALLOWED_CHAT_IDS: List[int] = SETTINGS.get("allowed_chat_ids", [])
ALLOWED_USER_IDS: List[int] = SETTINGS.get("allowed_user_ids", [])

# Remove the following functions and variables:
# - load_allowed_chat_ids
# - load_allowed_user_ids
# - ALLOWED_CHAT_IDS (the original declaration)
# - ALLOWED_USER_IDS (the original declaration)


def init_db():
    """Initialize the SQLite database and handle schema migrations."""
    conn = sqlite3.connect("message_history.db")
    c = conn.cursor()

    # Create a table to store schema version
    c.execute("""CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)""")

    # Get current schema version
    c.execute("SELECT version FROM schema_version")
    result = c.fetchone()
    current_version = result[0] if result else 0

    if current_version < 1:
        # Create the initial messages table
        c.execute(
            """CREATE TABLE IF NOT EXISTS messages
                     (chat_id INTEGER,
                      message_id INTEGER,
                      user_id INTEGER,
                      user_first_name TEXT,
                      user_last_name TEXT,
                      user_username TEXT,
                      message_text TEXT,
                      timestamp DATETIME,
                      is_forward BOOLEAN,
                      forward_from_name TEXT,
                      forward_from_id INTEGER,
                      forward_from_chat_id INTEGER,
                      forward_from_chat_title TEXT,
                      forward_date DATETIME,
                      PRIMARY KEY (chat_id, message_id))"""
        )

    if current_version < 2:
        # Add new columns for version 2
        c.execute("ALTER TABLE messages ADD COLUMN is_edit BOOLEAN")
        c.execute("ALTER TABLE messages ADD COLUMN edit_date DATETIME")
        c.execute("ALTER TABLE messages ADD COLUMN is_channel_post BOOLEAN")
        c.execute("ALTER TABLE messages ADD COLUMN channel_chat_id INTEGER")
        c.execute("ALTER TABLE messages ADD COLUMN channel_chat_title TEXT")
        c.execute("ALTER TABLE messages ADD COLUMN reply_to_message_id INTEGER")

        # Update schema version
        if current_version == 0:
            c.execute("INSERT INTO schema_version (version) VALUES (2)")
        else:
            c.execute("UPDATE schema_version SET version = 2")

    conn.commit()
    conn.close()


async def store_message_history(update: Update, _: CallbackContext) -> None:
    """Store comprehensive message data in the SQLite database."""
    message = (
        update.message
        or update.edited_message
        or update.channel_post
        or update.edited_channel_post
    )
    if not message:
        return

    chat_id = message.chat_id
    if ALLOWED_CHAT_IDS and chat_id not in ALLOWED_CHAT_IDS:
        return

    conn = sqlite3.connect("message_history.db")
    c = conn.cursor()

    is_forward = False
    forward_from_name = None
    forward_from_id = None
    forward_from_chat_id = None
    forward_from_chat_title = None
    forward_date = None

    # Check for forward information using hasattr
    if hasattr(message, "forward_from"):
        is_forward = message.forward_from is not None
        if is_forward:
            forward_from_name = (
                f"{message.forward_from.first_name} "
                "{message.forward_from.last_name or ''}".strip()
            )
            forward_from_id = message.forward_from.id
    elif hasattr(message, "forward_from_chat"):
        is_forward = message.forward_from_chat is not None
        if is_forward:
            forward_from_chat_id = message.forward_from_chat.id
            forward_from_chat_title = message.forward_from_chat.title

    if is_forward and hasattr(message, "forward_date"):
        forward_date = message.forward_date
    else:
        forward_date = message.date

    is_edit = bool(update.edited_message or update.edited_channel_post)
    is_channel_post = bool(update.channel_post or update.edited_channel_post)

    c.execute(
        """INSERT OR REPLACE INTO messages VALUES 
           (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chat_id,
            message.message_id,
            message.from_user.id if message.from_user else None,
            message.from_user.first_name if message.from_user else None,
            message.from_user.last_name if message.from_user else None,
            message.from_user.username if message.from_user else None,
            message.text or message.caption,
            datetime.now(),
            is_forward,
            forward_from_name,
            forward_from_id,
            forward_from_chat_id,
            forward_from_chat_title,
            forward_date,
            is_edit,
            message.edit_date,
            is_channel_post,
            message.chat.id if is_channel_post else None,
            message.chat.title if is_channel_post else None,
            message.reply_to_message.message_id if message.reply_to_message else None,
        ),
    )

    conn.commit()
    conn.close()


async def check_user_permission(update: Update) -> bool:
    """Check if the user is allowed to interact with the bot."""
    user_id = update.effective_user.id
    return not ALLOWED_USER_IDS or user_id in ALLOWED_USER_IDS


async def start(update: Update, _: CallbackContext) -> None:
    """Send a welcome message when the command /start is issued."""
    if not await check_user_permission(update):
        await update.message.reply_text(
            "Sorry, you are not authorized to use this bot."
        )
        return
    await update.message.reply_text(
        "Hello! Use /tldr <N> to get a summary of the last N messages."
    )


async def get_chat_id(update: Update, context: CallbackContext) -> None:
    """Respond with the chat ID of the current chat."""
    if not await check_user_permission(update):
        await update.message.reply_text(
            "Sorry, you are not authorized to use this bot."
        )
        return
    logging.info("Chat ID: %s", update.message.chat_id)
    if len(context.args) and context.args[0] == "hidden":
        return
    await update.message.reply_text(f"Chat ID: {update.message.chat_id}")


async def tldr(update: Update, context: CallbackContext) -> None:
    """Summarize recent chat messages."""
    if not await check_user_permission(update):
        await update.message.reply_text(
            "Sorry, you are not authorized to use this bot."
        )
        return

    chat_id = update.message.chat_id

    if chat_id not in ALLOWED_CHAT_IDS:
        await update.message.reply_text(
            "Sorry, this bot is not available in this chat."
        )
        return

    number_of_lines = get_number_of_lines(context.args)
    if number_of_lines is None:
        await update.message.reply_text("Usage: /tldr <number_of_lines>")
        return

    messages = get_recent_messages(chat_id, number_of_lines)
    logging.info("Found %d messages in chat %d", len(messages), chat_id)

    if not messages:
        await update.message.reply_text("No messages found to summarize.")
        return

    text_to_summarize = format_messages_for_summary(messages)

    if len(text_to_summarize) > 4000:
        await update.message.reply_text(
            "Too many messages to summarize. Please try a smaller number."
        )
        return
    try:
        summary = get_summary_from_openai(text_to_summarize)
        await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)
    except openai.APIError as e:
        logging.error("OpenAI API error: %s", e)
        await update.message.reply_text(
            "An error occurred while generating the summary. Please try again later."
        )
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        await update.message.reply_text(
            "An unexpected error occurred. Please try again later."
        )


def get_number_of_lines(args):
    """Extract and validate the number of lines from command arguments."""
    try:
        return int(args[0])
    except (IndexError, ValueError):
        return None


def get_recent_messages(chat_id: int, number_of_lines: int):
    """Fetch the specified number of recent messages from the database."""
    conn = sqlite3.connect("message_history.db")
    c = conn.cursor()
    c.execute(
        """SELECT user_first_name, user_last_name, message_text, is_forward, 
                 forward_from_name, forward_from_chat_title, reply_to_message_id,
                 message_id
                 FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?""",
        (chat_id, number_of_lines),
    )
    messages = c.fetchall()
    conn.close()
    return messages[::-1]  # Reverse the list to get oldest messages first


def format_messages_for_summary(messages):
    """Format messages into a single text string for summarization."""
    formatted_messages = []
    message_dict = {
        msg[7]: msg for msg in messages
    }  # Create a dict with message_id as key

    for msg in messages:
        user_name = f"{msg[0]} {msg[1] or ''}".strip()
        message_text = msg[2]
        is_forward = msg[3]
        forward_from_name = msg[4]
        forward_from_chat_title = msg[5]
        reply_to_message_id = msg[6]

        formatted_message = ""

        # Handle replies
        if reply_to_message_id and reply_to_message_id in message_dict:
            replied_msg = message_dict[reply_to_message_id]
            replied_user = f"{replied_msg[0]} {replied_msg[1] or ''}".strip()
            formatted_message += f"(In reply to {replied_user}) "

        # Handle forwards
        if is_forward:
            if forward_from_name:
                formatted_message += (
                    f"{user_name} forwarded from {forward_from_name}: {message_text}"
                )
            elif forward_from_chat_title:
                formatted_message += (
                    f"{user_name} forwarded from"
                    f"{forward_from_chat_title}: {message_text}"
                )
            else:
                formatted_message += f"{user_name} forwarded: {message_text}"
        else:
            formatted_message += f"{user_name}: {message_text}"

        formatted_messages.append(formatted_message)

    return "\n".join(formatted_messages)


def get_summary_from_openai(text_to_summarize):
    """Get a summary of the text using OpenAI's API."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text.",
            },
            {
                "role": "user",
                "content": "Summarize the following conversation from a chat platform. Provide a brief TLDR for the entire conversation or, if it aids clarity, an individual TLDR for each participant. The conversation includes replies and forwards. The output should be formatted in markdown without using any enclosing backtick characters. Ensure the TLDR language and tone mirror the original chat messages (e.g., if the input is in casual Japanese, the TLDR should be in casual Japanese). Do not include the original messages. The summary should be concise and capture the key information.\n\n"
                f"## chat messages:\n```{text_to_summarize}```\n\n",
            },
        ],
        max_tokens=4000,
    )

    return response.choices[0].message.content.strip()


def main() -> None:
    """Initialize and run the Telegram bot application."""
    init_db()  # Initialize the database
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("tldr", tldr))
    application.add_handler(CommandHandler("chat_id", get_chat_id))
    application.add_handler(
        MessageHandler(
            callback=store_message_history,
            filters=filters.TEXT
            | filters.CAPTION
            | filters.UpdateType.EDITED_MESSAGE
            | filters.UpdateType.CHANNEL_POST
            | filters.UpdateType.EDITED_CHANNEL_POST,
        )
    )

    try:
        application.run_polling()
    except Exception as e:
        logging.error("Error running the bot: %s", e)


if __name__ == "__main__":
    main()
