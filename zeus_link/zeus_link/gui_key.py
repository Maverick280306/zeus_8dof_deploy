"""
The GUI's key, kept in one file so the address stays the same across launches.

No ROS here: the `zeus` CLI and the unit tests use it without a ROS
environment. See gui_node.py for why there is a key at all.
"""

from __future__ import annotations

import os
import secrets
import socket
from typing import Optional

KEY_FILE = os.path.expanduser("~/.config/zeus/gui_key")


def stable_key(path: str = KEY_FILE) -> str:
    """The GUI key: read from path, or made once and saved there (mode 600)."""
    try:
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    except OSError:
        pass
    key = secrets.token_urlsafe(9)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key + "\n")
    return key


def gui_url(port: int = 8080, key: Optional[str] = None) -> str:
    return f"http://{socket.gethostname()}.local:{port}/?key={key or stable_key()}"
