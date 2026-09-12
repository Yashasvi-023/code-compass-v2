"""
database.py — All MongoDB operations for users and chats.
"""
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from datetime import datetime
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Connection (cached so it's only created once per Streamlit session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():

    uri = st.secrets["MONGO_URI"]

    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000
    )

    # FORCE a connection test
    client.admin.command("ping")

    return client["anti_code_compass"]


# ─────────────────────────────────────────────────────────────────────────────
# User operations
# ─────────────────────────────────────────────────────────────────────────────

def create_user(full_name: str, email: str, password_hash: str) -> dict | None:
    """
    Insert a new user. Returns the created user doc, or None if email exists.
    """
    db = get_db()
    try:
        result = db.users.insert_one({
            "full_name":     full_name,
            "email":         email.lower().strip(),
            "password_hash": password_hash,
            "created_at":    datetime.utcnow(),
        })
        return db.users.find_one({"_id": result.inserted_id})
    except DuplicateKeyError:
        return None  # email already exists


def find_user_by_email(email: str) -> dict | None:
    """Return a user doc by email, or None if not found."""
    db = get_db()
    return db.users.find_one({"email": email.lower().strip()})


# ─────────────────────────────────────────────────────────────────────────────
# Chat operations
# ─────────────────────────────────────────────────────────────────────────────

def create_chat(user_id: str, repo_url: str, repo_name: str) -> str:
    """Create a new chat session. Returns the new chat's string ID."""
    db = get_db()
    result = db.chats.insert_one({
        "user_id":    ObjectId(user_id),
        "repo_url":   repo_url,
        "repo_name":  repo_name,
        "created_at": datetime.utcnow(),
        "messages":   [],          # list of {role, content, timestamp}
    })
    return str(result.inserted_id)


def get_user_chats(user_id: str) -> list:
    """Return all chats for a user, newest first."""
    db = get_db()
    chats = db.chats.find(
        {"user_id": ObjectId(user_id)},
        {"messages": 0}            # don't load messages for the sidebar list
    ).sort("created_at", DESCENDING)
    return list(chats)


def get_chat_messages(chat_id: str) -> list:
    """Return just the messages array for a given chat."""
    db = get_db()
    chat = db.chats.find_one({"_id": ObjectId(chat_id)}, {"messages": 1})
    return chat["messages"] if chat else []


def get_chat(chat_id: str) -> dict | None:
    """Return the full chat document."""
    db = get_db()
    return db.chats.find_one({"_id": ObjectId(chat_id)})


def append_message(chat_id: str, role: str, content: str):
    """Append a single message to a chat's messages array."""
    db = get_db()
    db.chats.update_one(
        {"_id": ObjectId(chat_id)},
        {"$push": {"messages": {
            "role":      role,
            "content":   content,
            "timestamp": datetime.utcnow(),
        }}}
    )
