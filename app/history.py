from langchain_community.chat_message_histories import SQLChatMessageHistory
from app.config import DB_PATH, MEMORY_WINDOW_K


def get_session_history_db(session_id: str) -> SQLChatMessageHistory:
    """
    Return the SQLite-backed message history for a given session ID.
    Each session ID gets its own isolated history row in the DB.
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=DB_PATH,
    )


def get_recent_chat_history(session_id: str, k: int = MEMORY_WINDOW_K):
    """Return the last 2*k messages (k human + k AI turns) for display."""
    history = get_session_history_db(session_id)
    return history.messages[-(2 * k):]


def clear_session(session_id: str) -> None:
    """Delete all stored messages for the given session ID."""
    get_session_history_db(session_id).clear()
