"""
memory_store.py
---------------
Chat history store backed by MongoDB.

- Each session = one MongoDB document
- Fetches from MongoDB on every get_history() call
- Saves to MongoDB on every add_exchange() call
- Falls back to in-memory dict if MongoDB is unavailable

MongoDB document structure:
{
    "_id": "session_abc",           # session_id is the primary key
    "messages": [
        {"role": "human", "content": "What is this about?"},
        {"role": "ai",    "content": "This document is about..."}
    ],
    "updated_at": "2026-05-31T..."
}

.env setup:
    MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
    MONGO_DB=document_portal
    MONGO_COLLECTION=chat_history
"""

from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# MongoDB client — lazy import so app starts even if pymongo not installed
try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
from dotenv import load_dotenv
load_dotenv()  # ← add this before os.getenv calls

MONGO_URI        = os.getenv("MONGO_URI", "")
MONGO_DB         = os.getenv("MONGO_DB", "document_portal")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "chat_history")
MAX_HISTORY      = int(os.getenv("MEMORY_MAX_HISTORY", "20"))


class SessionMemoryStore:
    """
    MongoDB-backed chat history store.
    Falls back to in-memory dict if MongoDB is not configured.
    """

    def __init__(self):
        self._fallback: Dict[str, List[BaseMessage]] = {}
        self._collection = None

        if not MONGO_AVAILABLE:
            print("[MEMORY_STORE] pymongo not installed — using in-memory fallback")
            return

        if not MONGO_URI:
            print("[MEMORY_STORE] MONGO_URI not set — using in-memory fallback")
            return

        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")  # verify connection
            db = client[MONGO_DB]
            self._collection = db[MONGO_COLLECTION]
            # Index on session_id for fast lookups
            self._collection.create_index("session_id", unique=True, background=True)
            print(f"[MEMORY_STORE] Connected to MongoDB — db={MONGO_DB} collection={MONGO_COLLECTION}")
        except Exception as e:
            print(f"[MEMORY_STORE] MongoDB connection failed: {e} — using in-memory fallback")
            self._collection = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Fetch chat history for a session from MongoDB."""
        if self._collection is None:
            return self._fallback.get(session_id, [])

        try:
            doc = self._collection.find_one({"session_id": session_id})
            if not doc:
                return []
            return self._deserialize(doc.get("messages", []))
        except Exception as e:
            print(f"[MEMORY_STORE] get_history error: {e}")
            return self._fallback.get(session_id, [])

    def add_exchange(self, session_id: str, human_msg: str, ai_msg: str) -> None:
        """Append one Q/A turn and upsert to MongoDB."""
        # Get current history
        history = self.get_history(session_id)
        history.append(HumanMessage(content=human_msg))
        history.append(AIMessage(content=ai_msg))

        # Trim to max
        if len(history) > MAX_HISTORY:
            history = history[len(history) - MAX_HISTORY:]

        serialized = self._serialize(history)

        if self._collection is None:
            self._fallback[session_id] = history
            return

        try:
            self._collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "session_id": session_id,
                        "messages": serialized,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
                upsert=True,  # creates doc if not exists, updates if exists
            )
        except Exception as e:
            print(f"[MEMORY_STORE] add_exchange error: {e}")
            self._fallback[session_id] = history  # fallback to RAM

    def clear(self, session_id: str) -> None:
        """Delete a session's history from MongoDB."""
        if self._collection is None:
            self._fallback.pop(session_id, None)
            return
        try:
            self._collection.delete_one({"session_id": session_id})
        except Exception as e:
            print(f"[MEMORY_STORE] clear error: {e}")

    def clear_all(self) -> None:
        """Delete ALL sessions from MongoDB."""
        if self._collection is None:
            self._fallback.clear()
            return
        try:
            self._collection.delete_many({})
        except Exception as e:
            print(f"[MEMORY_STORE] clear_all error: {e}")

    def get_turn_count(self, session_id: str) -> int:
        return len(self.get_history(session_id)) // 2

    def session_exists(self, session_id: str) -> bool:
        return self.get_turn_count(session_id) > 0

    def list_sessions(self) -> List[str]:
        if self._collection is None:
            return list(self._fallback.keys())
        try:
            return [d["session_id"] for d in self._collection.find({}, {"session_id": 1})]
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    #  Serialization helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialize(messages: List[BaseMessage]) -> List[dict]:
        return [
            {
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": str(m.content),
            }
            for m in messages
        ]

    @staticmethod
    def _deserialize(raw: List[dict]) -> List[BaseMessage]:
        result = []
        for m in raw:
            if m.get("role") == "human":
                result.append(HumanMessage(content=m["content"]))
            else:
                result.append(AIMessage(content=m["content"]))
        return result


# ------------------------------------------------------------------ #
#  Module-level singleton                                              #
# ------------------------------------------------------------------ #
MEMORY_STORE = SessionMemoryStore()