"""Telegram bot for summarizing chat messages using LangChain."""

# pylint: disable=global-statement

import logging
import os

import openai
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)

import configs
from db import get_recent_messages, init_db, store_message_history

load_dotenv(find_dotenv())

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

client = openai.OpenAI()


def update_openai_key():
    """Update OpenAI API key when settings change."""
    global OPENAI_API_KEY
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or configs.get_setting(
        "openai_api_key"
    )
    if not OPENAI_API_KEY:
        logging.error("OpenAI API Key not found in environment or settings file")
    openai.api_key = OPENAI_API_KEY


def update_telegram_token():
    """Update Telegram Bot Token when settings change."""
    global TELEGRAM_BOT_TOKEN
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or configs.get_setting(
        "telegram_bot_token"
    )
    if not TELEGRAM_BOT_TOKEN:
        logging.error("Telegram Bot Token not found in environment or settings file")


# Custom handler to check settings before processing any update
async def settings_check_handler(_: Update, __: CallbackContext):
    """Check settings before processing any update."""
    # The actual checking is now done by the file watcher,
    # so this function mainly serves as a hook for any additional pre-processing
    return True


async def check_user_permission(update: Update) -> bool:
    """Check if the user is allowed to interact with the bot."""
    user_id = update.effective_user.id
    allowed_user_ids = configs.ALLOWED_USER_IDS
    return not allowed_user_ids or user_id in allowed_user_ids


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

    if configs.ALLOWED_CHAT_IDS and chat_id not in configs.ALLOWED_CHAT_IDS:
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

    if len(text_to_summarize) > 20000:
        await update.message.reply_text(
            "Too many messages to summarize. Please try a smaller number."
        )
        return
    try:
        summary = get_summary_from_langchain(text_to_summarize)
        await update.message.reply_text(summary)
    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        logging.error("LangChain error: %s", e)
        await update.message.reply_text(
            "An error occurred while generating the summary. Please try again later."
        )


def get_number_of_lines(args):
    """Extract and validate the number of lines from command arguments."""
    try:
        return int(args[0])
    except (IndexError, ValueError):
        return None


def format_messages_for_summary(messages):
    """Format messages into a single text string for summarization."""
    formatted_messages = []
    message_dict = {msg[7]: msg for msg in messages}

    for msg in messages:
        user_id = str(msg[8])
        contact = configs.get_contact(user_id)
        if contact:
            user_name = f"{contact['first_name']} {contact['last_name']}".strip()
        else:
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


def get_summary_from_langchain(text_to_summarize):
    """Get a summary of the text using LangChain with OpenAI's API."""
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.05, top_p=0.8)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a helpful assistant specialized in summarizing text conversations from chat platforms. Follow these specific instructions to create concise and informative summaries that capture the essence of the provided chat messages."
            ),
            HumanMessagePromptTemplate.from_template(
                "Summarize the following conversation from a chat platform. Choose between two summarization methods based on clarity and informativeness:\n1. Provide a single summary for the entire conversation if it consists mainly of a continuous thread.\n2. Provide individual summaries for each participant if the conversation involves a debate or opposing viewpoints.\n\n**Guidelines:**\n- **Format:** Present the summary in plain text without any special formatting or enclosing characters.\n- **Language and Tone:** Ensure the summary's language and tone mirror the original chat messages (e.g., if the input is in casual Japanese, the summary should be in casual Japanese). If unsure about the language, default to the non-English language.\n- **RTL Handling:** For right-to-left (RTL) languages, prepend each new line with a Right-To-Left Mark (‎\u200F) and avoid starting lines with left-to-right (LTR) characters.\n- **Content:** Focus on key information and main points of the conversation, including relevant context from replies and forwarded messages. Omit the original messages and the term 'summary' from your output.\n- **Perspective:** Always write the summary from a third-person view.\n- **Length:** Keep the summary concise while ensuring all crucial information is included.\n\n**Chat messages:**\n```\n{text_to_summarize}\n```\n\n**Summary:**\n"
            ),
        ]
    )

    messages = prompt.format_messages(text_to_summarize=text_to_summarize)
    response = chat.invoke(messages)
    return response.content.strip()


def main() -> None:
    """Initialize and run the Telegram bot application."""
    init_db()  # Initialize the database

    # Set up callback functions for configs
    configs.on_settings_changed = lambda: (update_openai_key(), update_telegram_token())
    configs.on_contacts_changed = lambda: logging.info("Contacts updated")

    # Initialize configs
    configs.initialize()

    # Initial update of OpenAI key and Telegram token
    update_openai_key()
    update_telegram_token()

    if not TELEGRAM_BOT_TOKEN:
        logging.error("Telegram Bot Token is missing. Cannot start the bot.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add the settings check handler to run before any other handler
    application.add_handler(
        MessageHandler(filters.ALL, settings_check_handler), group=-1
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("tldr", tldr))
    application.add_handler(CommandHandler("chat_id", get_chat_id))
    application.add_handler(
        MessageHandler(
            callback=lambda update, context: store_message_history(
                update, configs.ALLOWED_CHAT_IDS
            ),
            filters=filters.TEXT
            | filters.CAPTION
            | filters.UpdateType.EDITED_MESSAGE
            | filters.UpdateType.CHANNEL_POST
            | filters.UpdateType.EDITED_CHANNEL_POST,
        )
    )

    try:
        application.run_polling()
    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        logging.error("Error running the bot: %s", e)


if __name__ == "__main__":
    main()
