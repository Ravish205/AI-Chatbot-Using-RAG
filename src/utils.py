"""
Utility functions for text normalization, hashing, logging, and chat history export.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

from config import settings


def get_logger(name: str) -> logging.Logger:
    """Configures and returns a structured logger for the specified module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger


logger = get_logger(__name__)


def calculate_file_hash(file_input: Union[str, Path, bytes]) -> str:
    """
    Computes SHA-256 hash of a file or byte array for deduplication.
    
    Args:
        file_input: Path to file or raw bytes.
        
    Returns:
        Hexadecimal SHA-256 checksum string.
    """
    hasher = hashlib.sha256()
    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    elif isinstance(file_input, bytes):
        hasher.update(file_input)
    else:
        raise TypeError(f"Unsupported file_input type: {type(file_input)}")
    return hasher.hexdigest()


def calculate_text_hash(text: str) -> str:
    """Computes SHA-256 hash of a text string."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean_text(text: str) -> str:
    """
    Cleans and normalizes extracted text.
    - Strips null bytes and control characters (preserves newlines and tabs).
    - Normalizes unicode spaces.
    - Collapses excessive empty lines and repeated whitespace.
    """
    if not text:
        return ""
    # Remove null bytes
    cleaned = text.replace("\x00", "")
    # Normalize non-breaking spaces and uncommon whitespace
    cleaned = cleaned.replace("\u00a0", " ").replace("\u200b", "")
    # Replace control characters except tab, newline, carriage return
    cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    # Normalize Windows line endings
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive newlines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Collapse multiple spaces/tabs into a single space within lines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def sanitize_user_input(query: str, max_chars: int = 2000) -> str:
    """
    Sanitizes raw user queries before passing them into retrieval and prompt chains.
    Trims excessive length and strips control sequences.
    """
    if not query:
        return ""
    trimmed = query.strip()[:max_chars]
    # Remove null bytes or terminal escapes
    trimmed = trimmed.replace("\x00", "")
    trimmed = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", trimmed)
    return trimmed


def format_source_citation(metadata: Dict[str, Any]) -> str:
    """
    Formats metadata dictionary into a standardized citation string.
    Example: 'quarterly_report.pdf (Page 3)' or 'notes.txt (Chunk 1)'
    """
    filename = metadata.get("source_filename") or metadata.get("source", "Unknown Document")
    page = metadata.get("page_number")
    chunk_index = metadata.get("chunk_index")
    
    parts = [Path(str(filename)).name]
    if page is not None and int(page) > 0:
        parts.append(f"Page {page}")
    elif chunk_index is not None:
        parts.append(f"Section {int(chunk_index) + 1}")
        
    return " — ".join(parts)


def export_chat_history_txt(messages: List[Dict[str, Any]]) -> str:
    """Exports chat messages into a clean, human-readable text document."""
    lines = [
        "=" * 70,
        " AI Chatbot Conversation History",
        f" Exported At: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "=" * 70,
        "",
    ]
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        sources = msg.get("sources", [])
        
        lines.append(f"[{role}]:")
        lines.append(content)
        
        if sources:
            lines.append("\n  Sources Cited:")
            for s in sources:
                lines.append(f"  - {s.get('citation', 'Unknown')}")
        lines.append("-" * 70)
        lines.append("")
        
    return "\n".join(lines)


def export_chat_history_json(messages: List[Dict[str, Any]]) -> str:
    """Exports chat messages into formatted JSON."""
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_messages": len(messages),
        "messages": messages,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
