#!/usr/bin/env python3
"""
inference.py — Mandatory OpenEnv inference entrypoint.

Connects an LLM agent to the EmailTriageEnv via the FastAPI REST API.
The agent iterates through all three tasks (easy → medium → hard),
calling /reset and /step endpoints, and emitting structured logs.

Environment variables:
    API_BASE_URL  — LLM API base URL (e.g. https://api.openai.com/v1)
    MODEL_NAME    — Model identifier (e.g. gpt-4o, mistral-large)
    HF_TOKEN      — HuggingFace token used as the API key
    ENV_URL       — (optional) Base URL of the running environment
                    (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN: str = os.getenv("HF_TOKEN")
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")

TASKS: List[str] = ["easy", "medium", "hard"]
SESSION_ID: str = "inference-run"

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI client
# ──────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",
)

# ──────────────────────────────────────────────────────────────────────────────
# Agent prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an email triage expert working for a corporate support team. "
    "Your task is to classify incoming emails by urgency and route them to "
    "the correct team while drafting professional replies."
)


def build_user_prompt(obs: Dict[str, Any], last_reward: Optional[float] = None) -> str:
    """Build the user prompt from the current observation."""
    history = obs.get("previous_actions", [])
    history_str = "\n".join(history) if history else "(none)"
    reward_str = str(last_reward) if last_reward is not None else "N/A"

    return (
        "You are an email triage expert. Given the email below, respond ONLY "
        "with a JSON object matching exactly:\n"
        '{"urgency": "low|medium|high|critical", '
        '"team": "billing|technical|sales|hr|management|legal", '
        '"reply_draft": "...", "reasoning": "..."}\n\n'
        f"Email subject: {obs['subject']}\n"
        f"Email body: {obs['body']}\n"
        f"Sender: {obs['sender']}\n\n"
        f"Previous actions this episode: {history_str}\n"
        f"Last reward: {reward_str}\n\n"
        "Respond ONLY with the JSON object. No markdown, no explanation outside the JSON."
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON extraction helper
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_ACTION: Dict[str, str] = {
    "urgency": "medium",
    "team": "technical",
    "reply_draft": "Thank you for reaching out. We will look into this.",
    "reasoning": "default",
}


def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from LLM output.
    Falls back to DEFAULT_ACTION on any failure.
    """
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON block inside markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return dict(DEFAULT_ACTION)


# ──────────────────────────────────────────────────────────────────────────────
# Environment API helpers
# ──────────────────────────────────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    """Call POST /reset and return the observation."""
    resp = requests.post(
        f"{ENV_URL}/reset",
        params={"session_id": SESSION_ID},
        json={"task_id": task_id, "session_id": SESSION_ID},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    """Call POST /step and return the full StepResult."""
    resp = requests.post(
        f"{ENV_URL}/step",
        params={"session_id": SESSION_ID},
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# LLM call
# ──────────────────────────────────────────────────────────────────────────────

def call_llm(obs: Dict[str, Any], last_reward: Optional[float] = None) -> Dict[str, Any]:
    """Send the prompt to the LLM and parse the JSON action."""
    user_prompt = build_user_prompt(obs, last_reward)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        print(f"  [LLM ERROR] {e}", file=sys.stderr)
        raw = ""

    action = extract_json(raw)

    # Validate and clamp values
    valid_urgency = {"low", "medium", "high", "critical"}
    valid_teams = {"billing", "technical", "sales", "hr", "management", "legal"}

    if action.get("urgency") not in valid_urgency:
        action["urgency"] = "medium"
    if action.get("team") not in valid_teams:
        action["team"] = "technical"
    if not action.get("reply_draft") or len(str(action["reply_draft"]).strip()) < 10:
        action["reply_draft"] = "Thank you for reaching out. We will look into this."
    if not action.get("reasoning"):
        action["reasoning"] = "default"

    return action


# ──────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ──────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str) -> Dict[str, Any]:
    """Run a single task episode and return summary."""
    task_name = f"email_triage_{task_id}"
    print(f"[START] task={task_name} env=email-triage-env model={MODEL_NAME}")

    obs = env_reset(task_id)
    rewards: List[float] = []
    last_reward: Optional[float] = None
    total_steps = 0
    done = False
    error = None

    max_steps = obs.get("max_steps", 5)

    while not done and total_steps < max_steps:
        total_steps += 1
        try:
            action = call_llm(obs, last_reward)
            result = env_step(action)

            reward = result["reward"]
            done = result["done"]
            obs = result["observation"]
            last_reward = reward
            rewards.append(reward)

            print(
                f"[STEP] step={total_steps} "
                f"action={json.dumps(action)} "
                f"reward={reward} "
                f"done={done} "
                f"error=null"
            )

            # Early exit if we got a near-perfect score
            if done:
                break

        except Exception as e:
            error = str(e)
            print(
                f"[STEP] step={total_steps} "
                f"action=null "
                f"reward=0.0 "
                f"done=true "
                f"error={json.dumps(error)}"
            )
            done = True

    final_score = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
    success = error is None and len(rewards) > 0

    print(
        f"[END] success={str(success).lower()} "
        f"steps={total_steps} "
        f"score={final_score} "
        f"rewards={json.dumps(rewards)}"
    )

    return {
        "task": task_name,
        "success": success,
        "steps": total_steps,
        "score": final_score,
        "rewards": rewards,
        "error": error,
    }


def main() -> None:
    """Run inference across all tasks."""
    print("=" * 60)
    print(f"EmailTriageEnv Inference Runner")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  API Base : {API_BASE_URL}")
    print(f"  Env URL  : {ENV_URL}")
    print("=" * 60)
    print()

    results: List[Dict[str, Any]] = []

    for task_id in TASKS:
        try:
            result = run_task(task_id)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to run task {task_id}: {e}")
            results.append({
                "task": f"email_triage_{task_id}",
                "success": False,
                "steps": 0,
                "score": 0.0,
                "rewards": [],
                "error": str(e),
            })
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_score = 0.0
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['task']:<30} score={r['score']:.4f}  steps={r['steps']}")
        total_score += r["score"]

    avg_score = total_score / len(results) if results else 0.0
    print(f"\n  Overall average score: {avg_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
