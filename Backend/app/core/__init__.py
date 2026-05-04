"""
app/core/__init__.py

Core package — shared across the entire app.

    config.py      → All settings, read from .env, validated at startup
    exceptions.py  → Custom exception hierarchy for clean error handling
"""

from app.core.config import Settings, get_settings
from app.core.exceptions import AppException

__all__ = [
    "Settings",
    "get_settings",
    "AppException",
]
