#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# validate.sh — End-to-end validation for EmailTriageEnv
#
# Usage:
#   chmod +x scripts/validate.sh
#   ./scripts/validate.sh [ENV_URL]
#
# Defaults to http://localhost:7860 if no argument given.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_URL="${1:-http://localhost:7860}"
PASS=0
FAIL=0

green()  { printf "\033[32m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }

check() {
    local desc="$1"
    local result="$2"
    if [ "$result" = "0" ]; then
        green "  ✓ $desc"
        PASS=$((PASS + 1))
    else
        red "  ✗ $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " EmailTriageEnv Validation Suite"
echo " Target: $ENV_URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Health check ──────────────────────────────────────────────────────
echo ""
yellow "1. Health Check"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$ENV_URL/health")
check "GET /health returns 200" "$([ "$HTTP_CODE" = "200" ] && echo 0 || echo 1)"

HEALTH_BODY=$(curl -s "$ENV_URL/health")
echo "$HEALTH_BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null
check "Health response has status=ok" "$?"

# ── 2. Reset endpoint ───────────────────────────────────────────────────
echo ""
yellow "2. Reset Endpoint"

for TASK in easy medium hard; do
    RESET_RESP=$(curl -s -X POST "$ENV_URL/reset?session_id=validate_${TASK}" \
        -H "Content-Type: application/json" \
        -d "{\"task_id\": \"$TASK\", \"session_id\": \"validate_${TASK}\"}")
    
    echo "$RESET_RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'email_id' in d, 'missing email_id'
assert 'subject' in d, 'missing subject'
assert 'body' in d, 'missing body'
assert d['task_id'] == '$TASK', f'wrong task_id: {d[\"task_id\"]}'
assert d['step_number'] == 0, f'step not 0: {d[\"step_number\"]}'
" 2>/dev/null
    check "POST /reset task=$TASK returns valid observation" "$?"
done

# ── 3. Step endpoint ────────────────────────────────────────────────────
echo ""
yellow "3. Step Endpoint"

# Reset for step test
curl -s -X POST "$ENV_URL/reset?session_id=validate_step" \
    -H "Content-Type: application/json" \
    -d '{"task_id": "easy", "session_id": "validate_step"}' > /dev/null

STEP_RESP=$(curl -s -X POST "$ENV_URL/step?session_id=validate_step" \
    -H "Content-Type: application/json" \
    -d '{
        "urgency": "medium",
        "team": "billing",
        "reply_draft": "Thank you for reaching out. We will review your invoice and correct any errors.",
        "reasoning": "The email mentions an invoice discrepancy, which is a billing issue of medium urgency."
    }')

echo "$STEP_RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'observation' in d, 'missing observation'
assert 'reward' in d, 'missing reward'
assert 'done' in d, 'missing done'
assert 'info' in d, 'missing info'
assert 0.0 <= d['reward'] <= 1.0, f'reward out of range: {d[\"reward\"]}'
" 2>/dev/null
check "POST /step returns valid StepResult" "$?"

# Check reward is reasonable (not 0.0 for a decent answer)
echo "$STEP_RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['reward'] > 0.3, f'reward suspiciously low: {d[\"reward\"]}'
" 2>/dev/null
check "Step reward > 0.3 for correct-ish answer" "$?"

# ── 4. State endpoint ───────────────────────────────────────────────────
echo ""
yellow "4. State Endpoint"

STATE_RESP=$(curl -s "$ENV_URL/state?session_id=validate_step")
echo "$STATE_RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['session_id'] == 'validate_step', 'wrong session'
assert d['step_number'] >= 1, 'step should be >= 1 after a step'
assert len(d['reward_history']) >= 1, 'reward_history should have entries'
" 2>/dev/null
check "GET /state returns coherent state" "$?"

# ── 5. Session isolation ────────────────────────────────────────────────
echo ""
yellow "5. Session Isolation"

curl -s -X POST "$ENV_URL/reset?session_id=session_A" \
    -H "Content-Type: application/json" \
    -d '{"task_id": "easy", "session_id": "session_A"}' > /dev/null

curl -s -X POST "$ENV_URL/reset?session_id=session_B" \
    -H "Content-Type: application/json" \
    -d '{"task_id": "hard", "session_id": "session_B"}' > /dev/null

STATE_A=$(curl -s "$ENV_URL/state?session_id=session_A")
STATE_B=$(curl -s "$ENV_URL/state?session_id=session_B")

python3 -c "
import sys, json
a = json.loads('$STATE_A')
b = json.loads('$STATE_B')
assert a['task_id'] == 'easy', f'session A should be easy, got {a[\"task_id\"]}'
assert b['task_id'] == 'hard', f'session B should be hard, got {b[\"task_id\"]}'
" 2>/dev/null
check "Sessions A and B are isolated" "$?"

# ── 6. Episode done guard ───────────────────────────────────────────────
echo ""
yellow "6. Episode Boundary"

# Complete all steps of an easy episode (max_steps=3)
curl -s -X POST "$ENV_URL/reset?session_id=validate_done" \
    -H "Content-Type: application/json" \
    -d '{"task_id": "easy", "session_id": "validate_done"}' > /dev/null

for i in 1 2 3; do
    curl -s -X POST "$ENV_URL/step?session_id=validate_done" \
        -H "Content-Type: application/json" \
        -d '{
            "urgency": "low",
            "team": "sales",
            "reply_draft": "Thank you for your email. We will look into this matter for you.",
            "reasoning": "General response."
        }' > /dev/null 2>&1
done

# Attempting another step should return 400
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$ENV_URL/step?session_id=validate_done" \
    -H "Content-Type: application/json" \
    -d '{
        "urgency": "low",
        "team": "sales",
        "reply_draft": "Thank you for your email. We will look into this matter for you.",
        "reasoning": "General response."
    }')
check "Step after done returns 400" "$([ "$HTTP_CODE" = "400" ] && echo 0 || echo 1)"

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$FAIL" = "0" ]; then
    green " ALL $PASS TESTS PASSED"
else
    red " $FAIL FAILED / $((PASS + FAIL)) TOTAL"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit "$FAIL"
