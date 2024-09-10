"""Configuration manager for handling settings and contacts."""

# pylint: disable=not-callable,global-statement

import json
import logging
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Global variables
SETTINGS_FILE = "./settings.json"
CONTACTS_FILE = "./contacts.json"
SETTINGS = {}
ALLOWED_CHAT_IDS = []
ALLOWED_USER_IDS = []
CONTACTS = {}

# Callback functions to be set by the main application

#pylint: disable=invalid-name
on_settings_changed = None
on_contacts_changed = None
#pylint: enable=invalid-name

class FileChangeHandler(FileSystemEventHandler):
    """Handles file modification events for settings and contacts files."""

    def on_modified(self, event):
        if os.path.abspath(event.src_path) == os.path.abspath(SETTINGS_FILE):
            load_settings()
            if callable(on_settings_changed):
                on_settings_changed()
        elif os.path.abspath(event.src_path) == os.path.abspath(CONTACTS_FILE):
            load_contacts()
            if callable(on_contacts_changed):
                on_contacts_changed()


def setup_file_watcher():
    """Set up a file watcher for the settings and contacts files."""
    observer = Observer()
    handler = FileChangeHandler()
    directory = os.path.dirname(SETTINGS_FILE)
    observer.schedule(handler, path=directory, recursive=False)
    observer.start()
    logging.info("File watcher started for settings and contacts files")


def load_settings():
    """Load settings from the JSON file."""
    global SETTINGS, ALLOWED_CHAT_IDS, ALLOWED_USER_IDS
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            SETTINGS = json.load(f)
        ALLOWED_CHAT_IDS = SETTINGS.get("allowed_chat_ids", [])
        ALLOWED_USER_IDS = SETTINGS.get("allowed_user_ids", [])
        logging.info("Settings reloaded from %s", SETTINGS_FILE)
    except FileNotFoundError:
        logging.warning("Settings file not found: %s", SETTINGS_FILE)
    except json.JSONDecodeError:
        logging.error("Invalid JSON in settings file: %s", SETTINGS_FILE)


def load_contacts():
    """Load contacts from the JSON file into the global CONTACTS dictionary."""
    global CONTACTS
    try:
        with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
            contacts_list = json.load(f)
            CONTACTS = {str(contact["id"]): contact for contact in contacts_list}
        logging.info("Contacts reloaded from %s", CONTACTS_FILE)
    except FileNotFoundError:
        logging.warning("Contacts file not found: %s", CONTACTS_FILE)
    except json.JSONDecodeError:
        logging.error("Invalid JSON in contacts file: %s", CONTACTS_FILE)


def get_setting(key, default=None):
    """Get a setting value by key."""
    return SETTINGS.get(key, default)


def get_contact(user_id):
    """Get a contact by user ID."""
    return CONTACTS.get(str(user_id))


def initialize():
    """Initialize the configuration manager."""
    load_settings()
    load_contacts()
    setup_file_watcher()
