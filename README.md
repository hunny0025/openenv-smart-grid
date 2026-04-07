# 📧 EmailTriageEnv

> A real-world corporate email triage reinforcement learning environment for the **OpenEnv Hackathon**.  
> Agents learn to classify urgency, route emails to the correct team, and draft professional replies.

---

## 🎯 Motivation

Email triage is one of the most universally relevant and high-value tasks in knowledge work. Every company's support, operations, and management teams make hundreds of routing decisions daily — decisions that require **reading comprehension**, **contextual reasoning**, **urgency assessment**, and **professional communication**.

This environment captures that complexity across three difficulty tiers:
- **Easy** — Unambiguous single-issue classification  
- **Medium** — Misleading tone that masks real urgency (requires reading past the surface)  
- **Hard** — Multi-message threads where context *shifts* across messages (requires temporal reasoning)

The grading is **deterministic** (no LLM calls), **genuinely partial-credit** (adjacent urgency levels get partial marks), and **reproducible** across runs.

---

## 🔁 Environment Loop

```
┌──────────────┐         ┌────────────────┐         ┌─────────────────┐
│              │  POST   │                │  JSON   │                 │
│    Agent     │──/step──│  EmailTriageEnv│────────▶│   StepResult    │
│   (LLM)     │         │   (FastAPI)    │         │                 │
│              │◀────────│                │         │ • observation   │
│              │ reward  │                │         │ • reward ∈[0,1] │
│              │ + obs   │                │         │ • done          │
└──────────────┘         └────────────────┘         │ • info{}        │
       │                        ▲                   └─────────────────┘
       │     POST /reset        │
       └────────────────────────┘
```

---

## 📦 Action Space

| Field         | Type     | Values                                                  | Description                        |
|---------------|----------|----------------------------------------------------------|------------------------------------|
| `urgency`     | `enum`   | `low` · `medium` · `high` · `critical`                   | Urgency classification             |
| `team`        | `enum`   | `billing` · `technical` · `sales` · `hr` · `management` · `legal` | Routing destination          |
| `reply_draft` | `string` | 1–3 sentence professional reply                          | Draft response to sender           |
| `reasoning`   | `string` | Brief chain-of-thought                                   | Explanation of classification      |

---

## 👁️ Observation Space

| Field              | Type       | Description                                    |
|--------------------|------------|------------------------------------------------|
| `email_id`         | `string`   | Unique identifier for the email sample         |
| `subject`          | `string`   | Email subject line                             |
| `body`             | `string`   | Full email body (or assembled thread for hard)  |
| `sender`           | `string`   | Sender email address                           |
| `timestamp`        | `string`   | ISO-8601 timestamp                             |
| `task_id`          | `string`   | `easy` · `medium` · `hard`                     |
| `step_number`      | `int`      | Current step index (0-based)                   |
| `max_steps`        | `int`      | Maximum steps allowed                          |
| `previous_actions` | `string[]` | JSON-serialised prior actions this episode     |

---

## 📋 Task Descriptions

### Task 1 — Easy: Single-Issue Classification
- **Input:** One clear email with an obvious urgency and team  
- **Action:** Just urgency + team (reply optional but scored if present)  
- **Grader:** Exact-match urgency (0.50) + exact-match team (0.50) = **1.0 max**  
- **Max steps:** 3  
- **Example:** `"My invoice has the wrong amount"` → urgency=medium, team=billing

### Task 2 — Medium: Multi-Signal Triage with Reply
- **Input:** Ambiguous email where casual tone hides a critical issue  
- **Action:** Full `EmailAction` including a quality reply  
- **Grader:** Urgency (0.30) + team (0.30) + reply quality (0.40)  
  - Reply quality = keyword coverage (60%) + professionalism heuristic (40%)  
- **Max steps:** 5  
- **Example:** `"quick question about the API lol"` — actually a prod-down checkout failure → urgency=critical, team=technical

### Task 3 — Hard: Thread Context Reasoning
- **Input:** A 3-message email thread where urgency escalates mid-thread  
- **Action:** Agent must identify the CURRENT state after reading all messages  
- **Grader:** Thread context (0.20) + urgency shift detection (0.30) + final routing (0.30) + reply coherence (0.20)  
- **Max steps:** 7  
- **Example:** Thread starts as billing query → final message threatens legal proceedings → urgency=critical, team=legal

---

## ⚖️ Grading Logic

All grading is **deterministic and reproducible** — no LLM calls inside graders.

### Urgency Scoring (Partial Credit)
```
distance = |predicted_index - true_index|
score = max(0, 1.0 - distance × 0.4)
```
Scale: `low=0, medium=1, high=2, critical=3`  
Adjacent levels (e.g., high vs critical) → **0.6** partial credit  
Two apart (e.g., medium vs critical) → **0.2** partial credit

### Team Scoring
- Exact match → **1.0**
- Acceptable alternative → **0.5** (e.g., "management" often acceptable)
- Related category → **0.25**

### Reply Quality
- **Keyword coverage:** fraction of required keywords present  
- **Professionalism:** heuristic checking length, greetings, resolution phrases, tone  

### Reasoning Bonus
- Non-trivial reasoning (>20 chars) → **+0.05**

---

## 📊 Baseline Scores

| Task   | Baseline Score | Notes                                      |
|--------|---------------|----------------------------------------------|
| Easy   | 0.72          | Most models get urgency+team right           |
| Medium | 0.51          | Casual tone causes misclassification         |
| Hard   | 0.38          | Thread reasoning is genuinely challenging     |

---

## 🚀 Setup & Running

### Docker (Recommended)

```bash
# Build the image
docker build -t email-triage-env .

# Run the environment
docker run -p 7860:7860 email-triage-env

# Run inference against it
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o \
HF_TOKEN=your-api-key \
ENV_URL=http://localhost:7860 \
python inference.py
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment
uvicorn app.main:app --host 0.0.0.0 --port 7860

# Validate the environment (in another terminal)
chmod +x scripts/validate.sh
./scripts/validate.sh

# Run inference
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o \
HF_TOKEN=your-api-key \
python inference.py
```

### API Endpoints

| Method | Path      | Description                            |
|--------|-----------|----------------------------------------|
| GET    | `/health` | Health check (returns 200 immediately) |
| POST   | `/reset`  | Start a new episode                    |
| POST   | `/step`   | Submit an action                       |
| GET    | `/state`  | Current episode state snapshot         |
| GET    | `/docs`   | Interactive Swagger documentation      |

---

## 🔧 Environment Variables

| Variable       | Required | Default                      | Description                          |
|----------------|----------|------------------------------|--------------------------------------|
| `API_BASE_URL` | Yes      | `https://api.openai.com/v1`  | LLM API base URL                     |
| `MODEL_NAME`   | Yes      | `gpt-4o`                     | Model identifier                     |
| `HF_TOKEN`     | Yes      | —                            | API key / HuggingFace token          |
| `ENV_URL`      | No       | `http://localhost:7860`      | Environment base URL for inference   |
| `PORT`         | No       | `7860`                       | Uvicorn listen port (Docker)         |

---

## 🔐 Session Management

The environment handles concurrency and memory management through a session-based system:

- **Isolation**: Each agent should use a unique `session_id` query parameter for state isolation.
- **TTL (Time-To-Live)**: Inactive sessions are automatically reaped after **30 minutes** of inactivity.
- **Session Cap**: The server enforces a hard limit of **100 concurrent sessions** to prevent resource exhaustion.
- **Self-Cleaning**: A background reaper task runs every 60 seconds to prune expired sessions.
- **Persistence**: Calling `/reset` on an existing `session_id` cleanly replaces the old episode state without memory leaks.

---

## 🏗️ Architecture

```
email-triage-env/
├── Dockerfile              # Production container
├── openenv.yaml            # OpenEnv manifest
├── requirements.txt        # Python dependencies
├── inference.py            # LLM agent entrypoint
├── README.md               # This file
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI endpoints
│   ├── env.py              # Core EmailTriageEnv class
│   ├── models.py           # Pydantic schemas
│   ├── tasks.py            # Task registry & data loading
│   ├── graders.py          # Deterministic per-task graders
│   ├── reward.py           # Partial-credit reward function
│   └── data/
│       ├── emails_easy.json
│       ├── emails_medium.json
│       └── emails_hard.json
└── scripts/
    └── validate.sh         # End-to-end validation
```

---

## 📝 License

MIT
