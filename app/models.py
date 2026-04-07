"""
Pydantic models for EmailTriageEnv.
Strictly typed observation, action, and step result schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class EmailObservation(BaseModel):
    """What the agent sees at each step."""

    email_id: str = Field(..., description="Unique identifier for the email")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Full email body text")
    sender: str = Field(..., description="Sender email or name")
    timestamp: str = Field(..., description="ISO-8601 timestamp of the email")
    task_id: str = Field(..., description="Task difficulty level: easy | medium | hard")
    step_number: int = Field(..., ge=0, description="Current step index (0-based)")
    max_steps: int = Field(..., ge=1, description="Maximum allowed steps for this episode")
    previous_actions: List[str] = Field(
        default_factory=list,
        description="JSON-serialised representations of prior EmailActions in this episode",
    )

    model_config = {"json_schema_extra": {"example": {
        "email_id": "easy_001",
        "subject": "Wrong invoice amount",
        "body": "Hi, my invoice from last month has the wrong amount. Can you fix it? — John",
        "sender": "john.doe@example.com",
        "timestamp": "2024-03-15T09:30:00Z",
        "task_id": "easy",
        "step_number": 0,
        "max_steps": 3,
        "previous_actions": [],
    }}}


class EmailAction(BaseModel):
    """The structured action the agent must produce."""

    urgency: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Urgency classification of the email"
    )
    team: Literal["billing", "technical", "sales", "hr", "management", "legal"] = Field(
        ..., description="Team responsible for handling this email"
    )
    reply_draft: str = Field(
        ...,
        min_length=10,
        description="1-3 sentence professional reply draft to the sender",
    )
    reasoning: str = Field(
        ...,
        description="Brief chain-of-thought explaining classification decisions",
    )

    @field_validator("reply_draft")
    @classmethod
    def reply_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reply_draft cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reasoning cannot be empty or whitespace-only")
        return v.strip()

    model_config = {"json_schema_extra": {"example": {
        "urgency": "medium",
        "team": "billing",
        "reply_draft": (
            "Thank you for bringing this to our attention, John. "
            "Our billing team will review your invoice and correct the amount within 24 hours. "
            "You'll receive a confirmation email once the correction is processed."
        ),
        "reasoning": "Billing dispute with no service disruption; medium urgency, billing team.",
    }}}


class StepResult(BaseModel):
    """Result returned after each /step call."""

    observation: EmailObservation = Field(..., description="Updated observation after the action")
    reward: float = Field(..., ge=0.0, le=1.0, description="Reward signal in [0, 1]")
    done: bool = Field(..., description="Whether the episode is finished")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary diagnostic information (score breakdown, ground truth, etc.)",
    )


class ResetRequest(BaseModel):
    """Optional body for /reset — allows specifying task and session."""

    task_id: Optional[Literal["easy", "medium", "hard"]] = Field(
        None,
        description="Which task to load. If omitted, a random task is chosen.",
    )
    session_id: Optional[str] = Field(
        None,
        description="Session identifier for state isolation across concurrent callers.",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok"] = "ok"
    version: str = "1.0.0"
    environment: str = "email-triage-env"


class StateResponse(BaseModel):
    """Full state snapshot returned by GET /state."""

    session_id: str
    task_id: str
    step_number: int
    max_steps: int
    done: bool
    cumulative_reward: float
    current_observation: Optional[EmailObservation] = None
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    reward_history: List[float] = Field(default_factory=list)
