"""Session persistence and management."""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional

from .state import SessionState, CLIMode
from ..config.cli_config import CLIConfig, get_sessions_dir

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Session manager using SQLite for persistence.

    Sessions are stored in ~/.agent-swarm/sessions/sessions.db
    """

    def __init__(self, config: CLIConfig):
        """
        Initialize session manager.

        Args:
            config: CLI configuration
        """
        self.config = config
        self.db_path = get_sessions_dir(config) / "sessions.db"
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure database and tables exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    name TEXT,
                    working_directory TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_directory
                ON sessions(working_directory)
            """)
            conn.commit()

    def generate_id(self) -> str:
        """
        Generate a unique session ID.

        Format: {timestamp}_{short_uuid}

        Returns:
            Session ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"

    async def save(self, session: SessionState) -> None:
        """
        Save session to database.

        Args:
            session: Session state to save
        """
        session.updated_at = datetime.now()
        state_json = session.model_dump_json()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, name, working_directory, mode, created_at, updated_at, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.name,
                    session.working_directory,
                    session.mode.value if isinstance(session.mode, CLIMode) else session.mode,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    state_json,
                ),
            )
            conn.commit()

        logger.debug(f"Saved session {session.session_id}")

    async def load(self, session_id: str) -> Optional[SessionState]:
        """
        Load session from database.

        Args:
            session_id: Session ID to load

        Returns:
            SessionState if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()

        if row:
            try:
                state_dict = json.loads(row[0])
                return SessionState(**state_dict)
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}")
                return None

        return None

    async def load_latest(
        self,
        working_directory: Optional[str] = None,
    ) -> Optional[SessionState]:
        """
        Load the most recent session.

        Args:
            working_directory: Optional filter by working directory

        Returns:
            Most recent SessionState if found
        """
        with sqlite3.connect(self.db_path) as conn:
            if working_directory:
                cursor = conn.execute(
                    """
                    SELECT state_json FROM sessions
                    WHERE working_directory = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (working_directory,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT state_json FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """
                )
            row = cursor.fetchone()

        if row:
            try:
                state_dict = json.loads(row[0])
                return SessionState(**state_dict)
            except Exception as e:
                logger.error(f"Failed to load latest session: {e}")
                return None

        return None

    async def list_sessions(
        self,
        working_directory: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """
        List sessions with optional filters.

        Args:
            working_directory: Optional filter by working directory
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries
        """
        with sqlite3.connect(self.db_path) as conn:
            if working_directory:
                cursor = conn.execute(
                    """
                    SELECT session_id, name, working_directory, mode,
                           created_at, updated_at
                    FROM sessions
                    WHERE working_directory = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (working_directory, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT session_id, name, working_directory, mode,
                           created_at, updated_at
                    FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            rows = cursor.fetchall()

        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row[0],
                "name": row[1],
                "working_directory": row[2],
                "mode": row[3],
                "created_at": row[4],
                "updated_at": row[5],
            })

        return sessions

    async def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted session {session_id}")

        return deleted

    async def cleanup_old_sessions(
        self,
        max_age_days: int = 30,
        keep_named: bool = True,
    ) -> int:
        """
        Clean up old sessions.

        Args:
            max_age_days: Maximum age in days
            keep_named: Keep sessions with names

        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            if keep_named:
                cursor = conn.execute(
                    """
                    DELETE FROM sessions
                    WHERE updated_at < ? AND (name IS NULL OR name = '')
                    """,
                    (cutoff_iso,),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE updated_at < ?",
                    (cutoff_iso,),
                )
            conn.commit()
            deleted = cursor.rowcount

        logger.info(f"Cleaned up {deleted} old sessions")
        return deleted

    async def rename(self, session_id: str, name: str) -> bool:
        """
        Rename a session.

        Args:
            session_id: Session ID
            name: New name

        Returns:
            True if renamed, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE sessions SET name = ? WHERE session_id = ?",
                (name, session_id),
            )
            conn.commit()
            return cursor.rowcount > 0
