"""
Per-task graders for EmailTriageEnv.

Each grader returns a float in [0.0, 1.0].
All grading is deterministic — no LLM calls, only exact match / keyword / regex scoring.

Keyword tiers:
  - mandatory: Must be present for minimum pass. Each hit = equal weight. Missing any = penalty.
  - important: Expected in a good reply. Each hit = scaled weight.
  - bonus: Above-and-beyond domain specificity. Each hit = small extra credit.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Union


from server.models import EmailAction
from server.reward import URGENCY_LEVELS


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _urgency_score(predicted: str, expected: str) -> float:
    """Distance-based partial credit for urgency classification."""
    pred_idx = URGENCY_LEVELS.index(predicted)
    true_idx = URGENCY_LEVELS.index(expected)
    distance = abs(pred_idx - true_idx)
    return max(0.0, 1.0 - distance * 0.4)


def _team_score(predicted: str, expected: str, acceptable: List[str]) -> float:
    """Exact or acceptable match scoring for team routing."""
    if predicted == expected:
        return 1.0
    if predicted in acceptable:
        return 0.5
    return 0.0


def _tiered_keyword_score(text: str, keywords_spec: Union[Dict[str, List[str]], List[str]]) -> float:
    """
    Score reply text against a tiered keyword specification.

    Accepts either:
      - Legacy format: List[str] (treated as all-mandatory, flat scoring)
      - New format: {"mandatory": [...], "important": [...], "bonus": [...]}

    Weights:
      mandatory keywords → 50% of score (high penalty per miss)
      important keywords → 35% of score
      bonus keywords     → 15% of score

    Returns a float in [0.0, 1.0].
    """
    text_lower = text.lower()

    # Legacy fallback: flat list → treat all as mandatory
    if isinstance(keywords_spec, list):
        if not keywords_spec:
            return 1.0
        hits = sum(1 for kw in keywords_spec if kw.lower() in text_lower)
        return hits / len(keywords_spec)

    mandatory = keywords_spec.get("mandatory", [])
    important = keywords_spec.get("important", [])
    bonus = keywords_spec.get("bonus", [])

    score = 0.0

    # Mandatory (50%): each miss subtracts proportionally
    if mandatory:
        m_hits = sum(1 for kw in mandatory if kw.lower() in text_lower)
        score += (m_hits / len(mandatory)) * 0.50
    else:
        score += 0.50  # No mandatory defined = full credit

    # Important (35%)
    if important:
        i_hits = sum(1 for kw in important if kw.lower() in text_lower)
        score += (i_hits / len(important)) * 0.35
    else:
        score += 0.35

    # Bonus (15%)
    if bonus:
        b_hits = sum(1 for kw in bonus if kw.lower() in text_lower)
        score += (b_hits / len(bonus)) * 0.15
    else:
        score += 0.15

    return min(score, 1.0)


def _reply_professionalism(reply: str) -> float:
    """
    Heuristic professionalism score based on structural and tone markers.
    Returns a value in [0.0, 1.0].

    Components (4 × 0.25 = 1.0):
      1. Length: at least 15 words (substantive reply, not one-liner)
      2. Greeting/acknowledgment: opens professionally
      3. Resolution path: commits to concrete next step
      4. Clean tone: no unprofessional language
    """
    score = 0.0
    reply_lower = reply.lower()
    word_count = len(reply.split())

    # ── Length: substantive reply ──
    if word_count >= 25:
        score += 0.25
    elif word_count >= 15:
        score += 0.15
    elif word_count >= 8:
        score += 0.05

    # ── Greeting / acknowledgment ──
    greetings = [
        "thank you", "thanks for", "hi ", "hello", "dear ",
        "good morning", "good afternoon", "we appreciate",
        "appreciate you", "noted", "acknowledged",
    ]
    if any(g in reply_lower for g in greetings):
        score += 0.25

    # ── Resolution / next-step commitment ──
    resolution_phrases = [
        "we will", "we'll", "our team", "look into", "resolve",
        "investigate", "get back to you", "follow up", "reach out",
        "priorit", "escalat", "working on", "within", "ensure",
        "correct", "fix", "address", "assess", "review",
    ]
    if any(r in reply_lower for r in resolution_phrases):
        score += 0.25

    # ── Clean tone (no unprofessional markers) ──
    bad_markers = ["lol", "lmao", "bruh", "idk", "smh", "wtf", "omg", "tbh"]
    if not any(b in reply_lower for b in bad_markers):
        score += 0.25

    return min(score, 1.0)


def _thread_context_score(text: str, ground_truth: Dict[str, Any]) -> float:
    """
    Assess whether the agent demonstrates awareness of the thread context shift.
    Uses thread_context_keywords from ground truth — these are domain-specific
    terms that the agent should reference if it understood the thread evolution.
    """
    context_keywords: List[str] = ground_truth.get("thread_context_keywords", [])
    if not context_keywords:
        return 1.0

    text_lower = text.lower()
    hits = sum(1 for kw in context_keywords if kw.lower() in text_lower)
    return hits / len(context_keywords)


def _get_reply_keywords(ground_truth: Dict[str, Any]) -> Union[Dict[str, List[str]], List[str]]:
    """
    Extract reply_keywords from ground truth, supporting both legacy (list)
    and new (tiered dict) formats.
    """
    rk = ground_truth.get("reply_keywords", [])
    return rk


# ──────────────────────────────────────────────────────────────────────────────
# Task-specific graders
# ──────────────────────────────────────────────────────────────────────────────

def grade_easy(action: EmailAction, ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    Easy task grader — single-issue classification.

    Weights:
        urgency_match : 0.50
        team_match    : 0.50
    Total max: 1.0

    Reply is NOT scored for easy (classification only), but keyword coverage
    is tracked in the breakdown for diagnostics.
    """
    u = _urgency_score(action.urgency, ground_truth["urgency"])
    t = _team_score(action.team, ground_truth["team"], ground_truth.get("acceptable_teams", []))

    # Diagnostic only — not in total
    kw = _tiered_keyword_score(action.reply_draft, _get_reply_keywords(ground_truth))

    total = u * 0.50 + t * 0.50
    return {
        "total": round(total, 4),
        "urgency_score": round(u, 4),
        "team_score": round(t, 4),
        "reply_keyword_coverage_diagnostic": round(kw, 4),
    }


def grade_medium(action: EmailAction, ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    Medium task grader — multi-signal triage with reply quality.

    Weights:
        urgency_match  : 0.30
        team_match     : 0.30
        reply_quality  : 0.40  →  keyword tier score (60%) + professionalism (40%)
    Total max: 1.0
    """
    u = _urgency_score(action.urgency, ground_truth["urgency"])
    t = _team_score(action.team, ground_truth["team"], ground_truth.get("acceptable_teams", []))

    kw = _tiered_keyword_score(action.reply_draft, _get_reply_keywords(ground_truth))
    prof = _reply_professionalism(action.reply_draft)
    reply_quality = kw * 0.60 + prof * 0.40

    total = u * 0.30 + t * 0.30 + reply_quality * 0.40
    return {
        "total": round(total, 4),
        "urgency_score": round(u, 4),
        "team_score": round(t, 4),
        "reply_keyword_score": round(kw, 4),
        "reply_professionalism": round(prof, 4),
        "reply_quality_composite": round(reply_quality, 4),
    }


def grade_hard(action: EmailAction, ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    Hard task grader — multi-email thread reasoning.

    Weights:
        thread_context_awareness  : 0.20  (does reply + reasoning reference the shift?)
        urgency_shift_detection   : 0.30  (did the agent catch the urgency change?)
        correct_final_routing     : 0.30  (routed to the correct final team?)
        coherent_reply            : 0.20  (tiered keywords + professionalism)
    Total max: 1.0
    """
    # Thread context awareness — scored against both reply and reasoning
    combined_text = action.reply_draft + " " + action.reasoning
    ctx = _thread_context_score(combined_text, ground_truth)

    # Urgency detection — same partial-credit scale
    u = _urgency_score(action.urgency, ground_truth["urgency"])

    # Team routing
    t = _team_score(action.team, ground_truth["team"], ground_truth.get("acceptable_teams", []))

    # Reply coherence
    kw = _tiered_keyword_score(action.reply_draft, _get_reply_keywords(ground_truth))
    prof = _reply_professionalism(action.reply_draft)
    reply_coherence = kw * 0.60 + prof * 0.40

    total = ctx * 0.20 + u * 0.30 + t * 0.30 + reply_coherence * 0.20
    return {
        "total": round(total, 4),
        "thread_context_score": round(ctx, 4),
        "urgency_shift_score": round(u, 4),
        "routing_score": round(t, 4),
        "reply_keyword_score": round(kw, 4),
        "reply_professionalism": round(prof, 4),
        "reply_coherence_composite": round(reply_coherence, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

GRADER_MAP = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade(task_id: str, action: EmailAction, ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    Route to the correct grader by task_id and return score breakdown.
    The returned dict always has a "total" key in [0.0, 1.0].
    """
    grader_fn = GRADER_MAP.get(task_id)
    if grader_fn is None:
        raise ValueError(f"No grader registered for task_id={task_id!r}")
    return grader_fn(action, ground_truth)
