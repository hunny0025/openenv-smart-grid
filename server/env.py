"""
Core EmailTriageEnv class.

Manages episode state: current email, step counter, action history,
reward accumulation, and done flag. Stateless across episodes —
each /reset starts fresh.
"""

from __future__ import annotations

import json
import random
import uuid
from typing import Any, Dict, List, Optional

from server.graders import grade
from server.models import EmailAction, EmailObservation, StepResult
from server.reward import compute_reward
from server.tasks import get_task_config, list_task_ids, sample_email


class EmailTriageEnv:
    """
    Reinforcement-learning environment for corporate email triage.

    Lifecycle:
        1. reset(task_id)  →  EmailObservation
        2. step(action)    →  StepResult  (repeat until done)
        3. state()         →  full snapshot at any time
    """

    def __init__(self) -> None:
        self._task_id: str = ""
        self._email: Dict[str, Any] = {}
        self._ground_truth: Dict[str, Any] = {}
        self._step: int = 0
        self._max_steps: int = 3
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._action_history: List[Dict[str, Any]] = []
        self._reward_history: List[float] = []
        self._session_id: str = ""

    # ──────────────────────────────────────────────────────────────────────
    # reset
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        email_id: Optional[str] = None,
    ) -> EmailObservation:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : str or None
            "easy", "medium", or "hard". Random if omitted.
        session_id : str or None
            Caller-supplied session key. Auto-generated if omitted.
        email_id : str or None
            Pin a specific email. Random if omitted.

        Returns
        -------
        EmailObservation
        """
        if task_id is None:
            task_id = random.choice(list_task_ids())
        config = get_task_config(task_id)

        self._task_id = task_id
        self._max_steps = config["max_steps"]
        self._email = sample_email(task_id, email_id)
        self._ground_truth = self._email["ground_truth"]
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._action_history = []
        self._reward_history = []
        self._session_id = session_id or str(uuid.uuid4())

        return self._build_observation()

    # ──────────────────────────────────────────────────────────────────────
    # step
    # ──────────────────────────────────────────────────────────────────────

    def step(self, action: EmailAction) -> StepResult:
        """
        Execute one agent action and return the result.

        Raises
        ------
        RuntimeError
            If the episode is already done or has not been reset.
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call /reset to start a new episode."
            )

        self._step += 1

        # Compute reward from the reward module
        reward = compute_reward(
            action,
            self._ground_truth,
            step=self._step,
            max_steps=self._max_steps,
        )

        # Also get the structured grader breakdown for info
        grader_breakdown = grade(self._task_id, action, self._ground_truth)

        self._cumulative_reward += reward
        self._reward_history.append(reward)
        self._action_history.append(action.model_dump())

        # Episode ends when we hit max steps OR score perfectly
        done = self._step >= self._max_steps or grader_breakdown["total"] >= 0.99
        self._done = done

        obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=done,
            info={
                "grader_breakdown": grader_breakdown,
                "grader_total": grader_breakdown["total"],
                "cumulative_reward": round(self._cumulative_reward, 4),
                "step": self._step,
                "remaining_steps": max(0, self._max_steps - self._step),
                "task_id": self._task_id,
                "ground_truth_urgency": self._ground_truth["urgency"],
                "ground_truth_team": self._ground_truth["team"],
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # state
    # ──────────────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        """Return a full snapshot of current episode state."""
        return {
            "session_id": self._session_id,
            "task_id": self._task_id,
            "step_number": self._step,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "current_observation": (
                self._build_observation().model_dump() if self._email else None
            ),
            "action_history": self._action_history,
            "reward_history": self._reward_history,
        }

    # ──────────────────────────────────────────────────────────────────────
    # internals
    # ──────────────────────────────────────────────────────────────────────

    def _build_observation(self) -> EmailObservation:
        """Construct the observation the agent sees."""
        # For hard tasks, use the pre-assembled thread body
        body = self._email.get("body_for_agent", self._email.get("body", ""))

        return EmailObservation(
            email_id=self._email["email_id"],
            subject=self._email["subject"],
            body=body,
            sender=self._email["sender"],
            timestamp=self._email["timestamp"],
            task_id=self._task_id,
            step_number=self._step,
            max_steps=self._max_steps,
            previous_actions=[json.dumps(a) for a in self._action_history],
        )

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def session_id(self) -> str:
        return self._session_id
