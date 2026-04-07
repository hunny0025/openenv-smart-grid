# Product Requirements Document: EmailTriageEnv

**Product Name:** EmailTriageEnv  
**Description:** A real-world corporate email triage reinforcement learning (RL) environment for the OpenEnv Hackathon.  
**Version:** 1.0.0  
**Phase:** Complete / Production-Ready  

## 1. Executive Summary

**EmailTriageEnv** is an advanced, text-based RL environment designed for evaluating the reading comprehension, context tracking, and reasoning capabilities of Generative AI agents. 

Unlike traditional code-generation or robotic simulators, this environment tests an agent's ability to act as a Corporate Support or Triage Engineer. It feeds the agent realistic emails ranging from unambiguous queries to complex multi-message threads with shifting contexts. The agent must grade the urgency, route the ticket to the correct department, and draft professional replies. The environment is engineered specifically to stand out in the **OpenEnv Hackathon** by providing deterministic grading, non-binary partial credit, and scenarios that require "reading past the surface."

## 2. Problem Statement

Evaluating large language models (LLMs) and agents purely on simple Q&A or math benchmarks fails to capture their utility in real-world, high-ambiguity enterprise tasks. Triage routing is a core business problem that involves:
- **Tone mismatch:** An urgent issue written casually, or a trivial issue written aggressively.
- **Context shifts:** A thread that starts as a basic query and turns into a legal threat mid-way through.
- **Ambiguity:** Scenarios where multiple actions might be semi-correct, but one is optimal.

Most current textual environments use binary pass/fail grading, which punishes models harshly for minor deviations and does not accurately resemble real life where adjacent answers have partial value.

## 3. Product Goals & Objectives

1.  **Differentiated Complexity:** Present an environment that is unambiguously real-world and challenges the LLM on genuine NLP hurdles (sarcasm, casual tone, hidden urgency, thread history reasoning).
2.  **Robust Evaluation:** Use a fully deterministic grading pipeline (no LLM-in-the-loop for grading) utilizing semantic proximity matching and robust regex to score models fairly and consistently. 
3.  **Genuine Partial Credit:** Output rewards that reflect distance errors (e.g., classifying a `critical` bug as `high` urgency should yield a 0.6 partial score, not a 0.0 failure).
4.  **Hackathon Compliance:** Perfectly adhere to the OpenEnv protocol, supplying the mandated OpenAPI REST interfaces, an `openenv.yaml` schema, and a standardized CLI `inference.py` script.

## 4. Product Features and Scope

### 4.1. Observation Space
What the Agent sees at every step (`EmailObservation`):
- `email_id` (String): Unique identifier.
- `subject` (String): E-mail subject.
- `body` (String): Raw text or assembled thread history.
- `sender` (String): Email origin.
- `timestamp` (String): ISO-8601 formatting.
- `task_id` (String): Difficulty indicator.
- `step_number` & `max_steps`: Agent progress markers.
- `previous_actions` (List[String]): JSON representations of past decisions.

### 4.2. Action Space
What the Agent must produce (`EmailAction`):
- `urgency` (Enum): `[low, medium, high, critical]`
- `team` (Enum): `[billing, technical, sales, hr, management, legal]`
- `reply_draft` (String): A minimal 1-3 sentence professional response.
- `reasoning` (String): Brief textual explanation showing a Chain of Thought (CoT).

> [!IMPORTANT]  
> Actions are strictly validated via Pydantic. Output failing to cast back to JSON natively defaults to a conservative fallback to keep the evaluation from crashing out completely.

### 4.3. Difficulty Tiers (Tasks)
The environment provides three escalating tasks.

1.  **Task 1: Easy (Single-Issue Classification)**
    - Clear urgency, unambiguous department matching.
    - Graded on strict matches (Urgency: 50%, Team: 50%).
    - Max Steps: 3.
2.  **Task 2: Medium (Multi-Signal Triage)**
    - Misleading or casual tone masking a serious issue, requiring reading comprehension to identify the true priority.
    - Graded on Urgency (30%), Team (30%), Reply Quality (40%).
    - Max Steps: 5.
3.  **Task 3: Hard (Thread Context Reasoning)**
    - Multi-email threads where circumstances shift (e.g., technical problem -> SLA violation -> legal implications). 
    - The agent MUST correctly react to the *newest* information. 
    - Graded on Context Awareness (20%), Urgency Shift Detection (30%), Final Routing (30%), and Coherence (20%).
    - Max Steps: 7.

## 5. Technical Requirements

### 5.1. Tech Stack
- **Language**: Python 3.11+
- **API Framework**: FastAPI & Uvicorn
- **Validation**: Pydantic v2
- **Agent Integration**: OpenAI API SDK (for `inference.py`)
- **Deployment**: Docker containerization

### 5.2. Core Endpoints
- `POST /reset`: Initializes the environment and session state, loading a randomly (or explicitly) selected task.
- `POST /step`: Exits the agent's action JSON into the deterministic grading logic and updates the internal state.
- `GET /state`: Fetches the entire episode lifecycle memory arrays.
- `GET /health`: Ultra-lite endpoint designed to handle automated HF Spaces readiness pings.

### 5.3. Performance & Stability Rules
> [!WARNING]  
> **Latency Requirement**: The grading algorithm must return in `< 200ms` as evaluation tests might spam the system. Consequently, zero external LLMs are permitted during the parsing or grading phases inside the environment. 
> 
> **Statelessness via Isolation**: Standard HTTP endpoints must track `session_id` to allow multiple, simultaneous OpenEnv evaluations without state corruption.

## 6. Evaluation and Success Metrics (Baseline Assumptions)
To prove the distinct value of this environment, the scoring gradients must meaningfully separate top-tier models from weak models. Evaluated agents should yield metrics similar to:
- **Easy Baseline**: `~0.72` Average. (Basic NLP parsing handles this easily).
- **Medium Baseline**: `~0.51` Average. (Detects failures in tone-based nuance).
- **Hard Baseline**: `~0.38` Average. (Isolates models that can effectively aggregate temporal memory events). 
