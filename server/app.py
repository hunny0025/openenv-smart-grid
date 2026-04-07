"""
FastAPI application for EmailTriageEnv.

Endpoints:
    POST /reset   → Start a new episode, returns EmailObservation
    POST /step    → Submit an action, returns StepResult
    GET  /state   → Current episode state snapshot
    GET  /health  → Health check for HF Spaces / automated pings

Session management:
    - Sessions keyed by session_id query param (default: "default")
    - Sessions expire after SESSION_TTL_SECONDS of inactivity (default: 30 min)
    - Max MAX_SESSIONS concurrent sessions (default: 100)
    - /reset on existing session replaces it cleanly (no leak)
    - Background reaper runs every 60s to prune stale sessions
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from server.env import EmailTriageEnv
from server.models import (
    EmailAction,
    EmailObservation,
    HealthResponse,
    ResetRequest,
    StateResponse,
    StepResult,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SESSION_TTL_SECONDS: int = 30 * 60  # 30 minutes of inactivity → auto-expire
MAX_SESSIONS: int = 100             # Hard cap to prevent memory leaks
REAPER_INTERVAL_SECONDS: int = 60   # How often the background reaper runs

# ──────────────────────────────────────────────────────────────────────────────
# Application setup
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("email_triage_env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="EmailTriageEnv",
    description=(
        "A real-world corporate email triage RL environment. "
        "Agents classify urgency, route to the correct team, and draft professional replies."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Session store — in-memory, keyed by session_id, with TTL tracking
# ──────────────────────────────────────────────────────────────────────────────

# Each entry: (EmailTriageEnv, last_accessed_timestamp)
_sessions: Dict[str, Tuple[EmailTriageEnv, float]] = {}

DEFAULT_SESSION = "default"


def _touch(session_id: str) -> None:
    """Update the last-accessed timestamp for a session."""
    if session_id in _sessions:
        env, _ = _sessions[session_id]
        _sessions[session_id] = (env, time.time())


def _get_or_create_session(session_id: str) -> EmailTriageEnv:
    """
    Retrieve an existing session or create a new one.

    Raises HTTPException if the session cap is reached and a new
    session cannot be created.
    """
    if session_id in _sessions:
        _touch(session_id)
        return _sessions[session_id][0]

    # Enforce session cap
    if len(_sessions) >= MAX_SESSIONS:
        # Try to reap stale sessions first
        _reap_expired_sessions()
        if len(_sessions) >= MAX_SESSIONS:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Maximum concurrent sessions ({MAX_SESSIONS}) reached. "
                    "Try again later or reuse an existing session_id."
                ),
            )

    env = EmailTriageEnv()
    _sessions[session_id] = (env, time.time())
    logger.info("SESSION_CREATE session=%s total_active=%d", session_id, len(_sessions))
    return env


def _reap_expired_sessions() -> int:
    """Remove sessions that have been inactive beyond TTL. Returns count reaped."""
    now = time.time()
    expired = [
        sid for sid, (_, ts) in _sessions.items()
        if (now - ts) > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]
    if expired:
        logger.info("SESSION_REAP reaped=%d remaining=%d", len(expired), len(_sessions))
    return len(expired)


# ──────────────────────────────────────────────────────────────────────────────
# Background reaper task
# ──────────────────────────────────────────────────────────────────────────────

async def _reaper_loop() -> None:
    """Periodically prune stale sessions."""
    while True:
        await asyncio.sleep(REAPER_INTERVAL_SECONDS)
        _reap_expired_sessions()


@app.on_event("startup")
async def _start_reaper() -> None:
    asyncio.create_task(_reaper_loop())
    logger.info(
        "Session reaper started (TTL=%ds, interval=%ds, max=%d)",
        SESSION_TTL_SECONDS,
        REAPER_INTERVAL_SECONDS,
        MAX_SESSIONS,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Return 200 immediately for HF Spaces automated health pings."""
    return HealthResponse()


# ──────────────────────────────────────────────────────────────────────────────
# POST /reset
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=EmailObservation, tags=["environment"])
async def reset(
    body: Optional[ResetRequest] = None,
    session_id: Optional[str] = Query(None, description="Session isolation key"),
):
    """
    Start a new episode.

    Behaviour:
    - If the session_id already exists, the existing environment is RESET
      (not duplicated). This is safe to call repeatedly.
    - If session_id is omitted, the "default" session is used.
    - If task_id is omitted, a random task is selected.
    """
    task_id = None
    sid = session_id or DEFAULT_SESSION

    if body is not None:
        task_id = body.task_id
        if body.session_id:
            sid = body.session_id

    env = _get_or_create_session(sid)
    obs = env.reset(task_id=task_id, session_id=sid)
    _touch(sid)

    logger.info(
        "RESET session=%s task=%s email=%s max_steps=%d",
        sid, obs.task_id, obs.email_id, obs.max_steps,
    )
    return obs


# ──────────────────────────────────────────────────────────────────────────────
# POST /step
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/step", response_model=StepResult, tags=["environment"])
async def step(
    action: EmailAction,
    session_id: Optional[str] = Query(None, description="Session isolation key"),
):
    """
    Submit an agent action and receive the reward + next observation.

    Returns 400 if the episode is already done.
    Returns 404 if the session does not exist (call /reset first).
    """
    sid = session_id or DEFAULT_SESSION

    if sid not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"No active session for session_id={sid!r}. Call POST /reset first.",
        )

    env = _sessions[sid][0]
    _touch(sid)

    if env.is_done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call POST /reset to start a new episode.",
        )

    result = env.step(action)

    logger.info(
        "STEP session=%s step=%d reward=%.4f done=%s grader_total=%.4f",
        sid,
        result.info["step"],
        result.reward,
        result.done,
        result.info["grader_total"],
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# GET /state
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/state", response_model=StateResponse, tags=["environment"])
async def get_state(
    session_id: Optional[str] = Query(None, description="Session isolation key"),
):
    """
    Return the full state snapshot for the current session.

    Returns 404 if the session does not exist.
    """
    sid = session_id or DEFAULT_SESSION

    if sid not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"No active session for session_id={sid!r}. Call POST /reset first.",
        )

    env = _sessions[sid][0]
    _touch(sid)
    state_dict = env.state()

    return StateResponse(
        session_id=state_dict["session_id"],
        task_id=state_dict["task_id"],
        step_number=state_dict["step_number"],
        max_steps=state_dict["max_steps"],
        done=state_dict["done"],
        cumulative_reward=state_dict["cumulative_reward"],
        current_observation=(
            EmailObservation(**state_dict["current_observation"])
            if state_dict["current_observation"]
            else None
        ),
        action_history=state_dict["action_history"],
        reward_history=state_dict["reward_history"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Root
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["system"])
async def root():
    """Serve the Neural Terminal dashboard on the root endpoint."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    
    # Fallback if index.html is somehow missing
    return {
        "message": "EmailTriageEnv is running. Visit /docs for API documentation.",
        "endpoints": ["/health", "/reset", "/step", "/state", "/docs"],
        "session_config": {
            "default_session": DEFAULT_SESSION,
            "ttl_seconds": SESSION_TTL_SECONDS,
            "max_sessions": MAX_SESSIONS,
        },
    }
