"""
Task definitions for EmailTriageEnv.
Loads email data from JSON files and exposes task configs keyed by difficulty.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent / "data"

# ──────────────────────────────────────────────
# Task registry
# ──────────────────────────────────────────────

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "easy": {
        "name": "Single-issue classification",
        "description": (
            "The agent receives a single email with an unambiguous issue. "
            "It must classify urgency and route to the correct team."
        ),
        "difficulty": "easy",
        "max_steps": 3,
        "data_file": "emails_easy.json",
    },
    "medium": {
        "name": "Multi-signal triage with reply",
        "description": (
            "The agent receives an ambiguous email where tone and content diverge. "
            "It must look past surface signals, classify correctly, and draft "
            "a quality reply."
        ),
        "difficulty": "medium",
        "max_steps": 5,
        "data_file": "emails_medium.json",
    },
    "hard": {
        "name": "Thread context reasoning",
        "description": (
            "The agent receives a 3-message email thread where the urgency "
            "and responsible team shift across messages. It must reason over "
            "the full thread to determine the CURRENT situation and respond."
        ),
        "difficulty": "hard",
        "max_steps": 7,
        "data_file": "emails_hard.json",
    },
}


# ──────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────

_cache: Dict[str, List[Dict[str, Any]]] = {}


def _load_emails(task_id: str) -> List[Dict[str, Any]]:
    """Load and cache email data for a given task difficulty."""
    if task_id in _cache:
        return _cache[task_id]

    config = TASK_CONFIG.get(task_id)
    if config is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Choose from {list(TASK_CONFIG)}")

    data_path = DATA_DIR / config["data_file"]
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        emails = json.load(f)

    _cache[task_id] = emails
    return emails


def get_task_config(task_id: str) -> Dict[str, Any]:
    """Return the config dict for a task (name, description, max_steps, etc.)."""
    if task_id not in TASK_CONFIG:
        raise ValueError(f"Unknown task_id: {task_id!r}. Choose from {list(TASK_CONFIG)}")
    return TASK_CONFIG[task_id]


def get_email_pool(task_id: str) -> List[Dict[str, Any]]:
    """Return full pool of emails for a given task difficulty."""
    return _load_emails(task_id)


def sample_email(task_id: str, email_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Pick one email from the pool.
    If *email_id* is provided, return that specific email.
    Otherwise, choose uniformly at random.
    """
    pool = _load_emails(task_id)
    if email_id is not None:
        for e in pool:
            if e["email_id"] == email_id:
                return e
        raise ValueError(
            f"email_id {email_id!r} not found in {task_id} pool. "
            f"Available: {[e['email_id'] for e in pool]}"
        )
    return random.choice(pool)


def list_task_ids() -> List[str]:
    """Return available task IDs in difficulty order."""
    return ["easy", "medium", "hard"]
