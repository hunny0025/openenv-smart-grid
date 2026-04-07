"""
Reward function for EmailTriageEnv.

Computes a float reward in [0.0, 1.0] with *genuine* partial credit:
  - Adjacent urgency levels get partial credit, not zero.
  - Related team categories earn partial marks.
  - Reply quality uses tiered keyword scoring (mandatory/important/bonus).
  - Non-trivial reasoning earns a small bonus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from server.models import EmailAction


# Ordered urgency scale — distance determines partial credit
URGENCY_LEVELS: List[str] = ["low", "medium", "high", "critical"]

# Teams that are semantically "adjacent" for partial-credit purposes
RELATED_TEAMS: Dict[str, List[str]] = {
    "billing": ["management", "sales"],
    "technical": ["management"],
    "sales": ["management", "billing"],
    "hr": ["management", "legal"],
    "management": ["hr", "legal", "billing", "technical", "sales"],
    "legal": ["management", "hr"],
}


def _tiered_keyword_score(text: str, keywords_spec: Union[Dict[str, List[str]], List[str]]) -> float:
    """
    Score reply text against a tiered keyword specification.

    Accepts either:
      - Legacy format: List[str] → flat, equal-weight scoring
      - New format: {"mandatory": [...], "important": [...], "bonus": [...]}

    Tier weights: mandatory=50%, important=35%, bonus=15%.
    """
    text_lower = text.lower()

    if isinstance(keywords_spec, list):
        if not keywords_spec:
            return 1.0
        hits = sum(1 for kw in keywords_spec if kw.lower() in text_lower)
        return hits / len(keywords_spec)

    mandatory = keywords_spec.get("mandatory", [])
    important = keywords_spec.get("important", [])
    bonus = keywords_spec.get("bonus", [])

    score = 0.0

    if mandatory:
        m_hits = sum(1 for kw in mandatory if kw.lower() in text_lower)
        score += (m_hits / len(mandatory)) * 0.50
    else:
        score += 0.50

    if important:
        i_hits = sum(1 for kw in important if kw.lower() in text_lower)
        score += (i_hits / len(important)) * 0.35
    else:
        score += 0.35

    if bonus:
        b_hits = sum(1 for kw in bonus if kw.lower() in text_lower)
        score += (b_hits / len(bonus)) * 0.15
    else:
        score += 0.15

    return min(score, 1.0)


def compute_reward(
    action: EmailAction,
    ground_truth: Dict[str, Any],
    step: int,
    max_steps: int,
) -> float:
    """
    Compute a reward in [0.0, 1.0].

    Weight breakdown:
        urgency  : 0.35  (partial credit for adjacent levels)
        team     : 0.35  (exact = full, acceptable or related = partial)
        reply    : 0.25  (tiered keyword coverage in reply_draft)
        reasoning: 0.05  (bonus for non-trivial chain-of-thought)

    Parameters
    ----------
    action : EmailAction
        The agent's predicted action.
    ground_truth : dict
        Expected answer including urgency, team, acceptable_teams, reply_keywords.
    step : int
        Current step number (0-based).
    max_steps : int
        Maximum steps allowed for this episode.

    Returns
    -------
    float
        Clipped reward in [0.0, 1.0].
    """
    reward = 0.0

    # ── 1. Urgency scoring (0.35) ────────────────────────────────────────
    pred_idx = URGENCY_LEVELS.index(action.urgency)
    true_idx = URGENCY_LEVELS.index(ground_truth["urgency"])
    distance = abs(pred_idx - true_idx)
    urgency_score = max(0.0, 1.0 - distance * 0.4)
    reward += urgency_score * 0.35

    # ── 2. Team scoring (0.35) ───────────────────────────────────────────
    true_team: str = ground_truth["team"]
    acceptable: List[str] = ground_truth.get("acceptable_teams", [])

    if action.team == true_team:
        reward += 0.35
    elif action.team in acceptable:
        reward += 0.20
    elif action.team in RELATED_TEAMS.get(true_team, []):
        reward += 0.10

    # ── 3. Reply quality scoring (0.25) — tiered keywords ────────────────
    reply_keywords = ground_truth.get("reply_keywords", [])
    keyword_score = _tiered_keyword_score(action.reply_draft.lower(), reply_keywords)
    reward += keyword_score * 0.25

    # ── 4. Reasoning bonus (0.05) ────────────────────────────────────────
    if len(action.reasoning.strip()) > 20:
        reward += 0.05

    # ── Clamp to [0.0, 1.0] ─────────────────────────────────────────────
    return round(min(max(reward, 0.0), 1.0), 4)
