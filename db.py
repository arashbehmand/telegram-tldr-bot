"""Database operations for storing and retrieving message history."""

import sqlite3
from datetime import datetime
import logging
from telegram import Update


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
        logging.warning("Creating the initial messages table")
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
        logging.warning("Adding new columns for version 2")
        c.execute("ALTER TABLE messages ADD COLUMN is_edit BOOLEAN")
        c.execute("ALTER TABLE messages ADD COLUMN edit_date DATETIME")
        c.execute("ALTER TABLE messages ADD COLUMN is_channel_post BOOLEAN")
        c.execute("ALTER TABLE messages ADD COLUMN channel_chat_id INTEGER")
        c.execute("ALTER TABLE messages ADD COLUMN channel_chat_title TEXT")
        c.execute("ALTER TABLE messages ADD COLUMN reply_to_message_id INTEGER")

        # Update schema version
        logging.warning("Inserting new schema")
        if current_version == 0:
            c.execute("INSERT INTO schema_version (version) VALUES (2)")
        else:
            c.execute("UPDATE schema_version SET version = 2")

    conn.commit()
    conn.close()


async def store_message_history(update: Update, allowed_chat_ids: list) -> None:
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
    if allowed_chat_ids and chat_id not in allowed_chat_ids:
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
    logging.info("Message history updated.")


def get_recent_messages(chat_id: int, number_of_lines: int):
    """Fetch the specified number of recent messages from the database."""
    conn = sqlite3.connect("message_history.db")
    c = conn.cursor()
    c.execute(
        """SELECT user_first_name, user_last_name, message_text, is_forward, 
                 forward_from_name, forward_from_chat_title, reply_to_message_id,
                 message_id, user_id
                 FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?""",
        (chat_id, number_of_lines),
    )
    messages = c.fetchall()
    conn.close()
    return messages[::-1]  # Reverse the list to get oldest messages first
