import json
import os
from typing import Optional, Dict, Any

from .utils import ensure_dir, safe_json_dump, now_iso

DEFAULT_STORAGE_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage")


class StorageError(Exception):
    """Custom storage error."""


class BaseStorage:
    """Abstract storage interface for session data and files."""

    # PUBLIC_INTERFACE
    def save_session_note(self, session_id: str, note: Dict[str, Any]) -> None:
        """Save a structured note for a session."""
        raise NotImplementedError

    # PUBLIC_INTERFACE
    def get_latest_note(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the latest structured note for a session."""
        raise NotImplementedError

    # PUBLIC_INTERFACE
    def save_file(self, session_id: str, filename: str, content: bytes) -> str:
        """Save a file attached to a session. Returns absolute path or storage id."""
        raise NotImplementedError

    # PUBLIC_INTERFACE
    def list_files(self, session_id: str) -> Dict[str, Any]:
        """List files for a session."""
        raise NotImplementedError


class LocalStorage(BaseStorage):
    """
    Local filesystem storage for sessions and files.
    Directory structure:
    storage/
      sessions/{session_id}/notes.jsonl
      sessions/{session_id}/files/{filename}
    """

    def __init__(self, root: Optional[str] = None):
        self.root = root or DEFAULT_STORAGE_ROOT

    def _session_dir(self, session_id: str) -> str:
        return os.path.join(self.root, "sessions", session_id)

    def _files_dir(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), "files")

    def _notes_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), "notes.jsonl")

    def save_session_note(self, session_id: str, note: Dict[str, Any]) -> None:
        ensure_dir(self._session_dir(session_id))
        path = self._notes_path(session_id)
        note = dict(note)
        note.setdefault("timestamp", now_iso())
        with open(path, "a", encoding="utf-8") as f:
            f.write(safe_json_dump(note) + "\n")

    def get_latest_note(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self._notes_path(session_id)
        if not os.path.exists(path):
            return None
        last_line = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if last_line:
            try:
                return json.loads(last_line)
            except json.JSONDecodeError:
                return None
        return None

    def save_file(self, session_id: str, filename: str, content: bytes) -> str:
        ensure_dir(self._files_dir(session_id))
        dest = os.path.join(self._files_dir(session_id), filename)
        with open(dest, "wb") as f:
            f.write(content)
        return dest

    def list_files(self, session_id: str) -> Dict[str, Any]:
        dir_ = self._files_dir(session_id)
        files = []
        if os.path.isdir(dir_):
            for name in os.listdir(dir_):
                p = os.path.join(dir_, name)
                if os.path.isfile(p):
                    files.append({"filename": name, "path": p})
        return {"count": len(files), "files": files}


class OneDriveStorage(BaseStorage):
    """
    Placeholder OneDrive storage.
    In a real implementation, integrate Microsoft Graph API using OAuth credentials stored in environment.
    Here, we simulate availability via an env flag; otherwise raise to trigger fallback.
    """

    def __init__(self):
        self.enabled = os.getenv("ONEDRIVE_ENABLED", "false").lower() == "true"

    def _ensure_enabled(self):
        if not self.enabled:
            raise StorageError("OneDrive not enabled")

    def save_session_note(self, session_id: str, note: Dict[str, Any]) -> None:
        self._ensure_enabled()
        # Simulate success without actual remote IO.
        return

    def get_latest_note(self, session_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_enabled()
        return None

    def save_file(self, session_id: str, filename: str, content: bytes) -> str:
        self._ensure_enabled()
        return f"onedrive://{session_id}/{filename}"

    def list_files(self, session_id: str) -> Dict[str, Any]:
        self._ensure_enabled()
        return {"count": 0, "files": []}


# PUBLIC_INTERFACE
class HybridStorage(BaseStorage):
    """
    Hybrid storage that prefers OneDrive and falls back to Local.
    """

    def __init__(self):
        self.remote = OneDriveStorage()
        self.local = LocalStorage()

    def save_session_note(self, session_id: str, note: Dict[str, Any]) -> None:
        try:
            self.remote.save_session_note(session_id, note)
        except Exception:
            self.local.save_session_note(session_id, note)

    def get_latest_note(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            note = self.remote.get_latest_note(session_id)
            if note:
                return note
        except Exception:
            pass
        return self.local.get_latest_note(session_id)

    def save_file(self, session_id: str, filename: str, content: bytes) -> str:
        try:
            return self.remote.save_file(session_id, filename, content)
        except Exception:
            return self.local.save_file(session_id, filename, content)

    def list_files(self, session_id: str) -> Dict[str, Any]:
        try:
            data = self.remote.list_files(session_id)
            if data and data.get("count", 0) > 0:
                return data
        except Exception:
            pass
        return self.local.list_files(session_id)
