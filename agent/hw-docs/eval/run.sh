#!/usr/bin/env bash
# Cold-session runner — issue #25 — and the driver of the #26 baseline control.
#
# One fresh `claude -p` session per golden-set question, per arm:
#   with     the repo + corpus as they stand — the layer being measured
#   without  same repo, layer unwired: no AGENTS.md hardware block, no
#            corpus (INDEX.md stays on disk, referenced nowhere)
# Arms are built by arms.py from a git export of the measured rev, so the
# operator's working tree is never touched. Cold means cold: --no-session-
# persistence, and an isolated CLAUDE_CONFIG_DIR so the operator's plugins,
# hooks, skills and MCP servers stay out of the measured sessions (they are
# not part of the kit) while credentials are copied in for auth.
#
# Answer-key hygiene: arms.py removes agent/hw-docs/{eval,explore}/ from
# BOTH arms (the golden set is the answer key; explore/ is a full second
# corpus copy from the #27 spike). score.py flags any session that opened
# the key anyway — a clean run has zero contaminated rows.
#
# Scoring (score.py): routed/searched from tool use, citations through the
# #29 structural grader against the REAL corpus for both arms (a bare
# model's recited pages get checked against reality), verdicts via an LLM
# judge blind to which arm it is reading, per-category scoreboard + the
# with-vs-without delta table.
#
# Usage:
#   ./run.sh [--arms with,without] [--ids id1,id2] [--parallel 3]
#            [--timeout 900] [--retries 2] [--label NAME] [--rev REF]
#            [--dry-run] [--skip-lint] [--no-judge]
#
# Costs real tokens: ~27 fresh sessions per arm. Run the full set when
# AGENTS.md or INDEX.md changes; sample a few --ids otherwise (#25).
#
# Blocked on auth? `claude -p` needs a logged-in CLI: run `claude` once
# interactively and /login, then re-run. The runner checks up front.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # agent/hw-docs/eval
ROOT="$(cd "$HERE/../../.." && pwd)"           # repo root
HW="$ROOT/agent/hw-docs"

ARMS="with,without"
IDS=""
PARALLEL=3
TIMEOUT_S=900
RETRIES=2
LABEL="$(date +%Y-%m-%d)-run"
REV="HEAD"
DRY_RUN=0
SKIP_LINT=0
NO_JUDGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) ARMS="$2"; shift 2 ;;
    --ids) IDS="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --retries) RETRIES="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --rev) REV="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-lint) SKIP_LINT=1; shift ;;
    --no-judge) NO_JUDGE=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

CLAUDE_BIN="${CLAUDE_BIN:-claude}"
SCRATCH="${HW_EVAL_HOME:-${TMPDIR:-/tmp}}/hw-eval-$LABEL"
RESULTS="$HERE/results/$LABEL"
PYTHON="${PYTHON:-python}"

say() { printf '\033[1m[run]\033[0m %s\n' "$*"; }

# --- pre-flight ---------------------------------------------------------------

if [[ "$ARMS" == *with,* || "$ARMS" == *",with" || "$ARMS" == "with" ]] && [[ $SKIP_LINT -eq 0 ]]; then
  say "pre-flight: corpus lint (the With arm is measured against this exact state)"
  (cd "$ROOT" && "$PYTHON" "$HW/lint.py" --offline) >/dev/null \
    || { echo "lint.py --offline failed — fix the corpus before measuring" >&2; exit 1; }
fi

CRED_SOURCE="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/.credentials.json"
if [[ $DRY_RUN -eq 0 && ! -f "$CRED_SOURCE" ]]; then
  echo "no claude credentials at $CRED_SOURCE" >&2
  echo "run \`claude\` interactively once and /login, then re-run" >&2
  exit 1
fi
command -v "$CLAUDE_BIN" >/dev/null || { echo "claude CLI not found" >&2; exit 1; }

# --- scratch + isolated claude config -----------------------------------------

say "scratch: $SCRATCH"
mkdir -p "$SCRATCH"/{transcripts,meta,config}

# Isolated config: no operator plugins/hooks/skills/MCP in measured sessions.
# Model + effort are pinned from the operator's settings so the run is
# reproducible; credentials copied in for auth.
CONFIG_DIR="$SCRATCH/config"
"$PYTHON" - "$HOME/.claude/settings.json" "$CONFIG_DIR/settings.json" <<'PY'
import json, sys
from pathlib import Path
src, dst = sys.argv[1], sys.argv[2]
user = {}
p = Path(src)
if p.is_file():
    user = json.loads(p.read_text(encoding="utf-8"))
out = {"hasCompletedOnboarding": True}
for k in ("model", "effortLevel"):
    if k in user:
        out[k] = user[k]
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(out, indent=2), encoding="utf-8")
PY
if [[ $DRY_RUN -eq 0 ]]; then
  cp "$CRED_SOURCE" "$CONFIG_DIR/.credentials.json"
fi
# Node needs a Windows-form env var even under Git Bash
export CLAUDE_CONFIG_DIR="$(cygpath -w "$CONFIG_DIR" 2>/dev/null || echo "$CONFIG_DIR")"

# --- arms ---------------------------------------------------------------------

say "building arms from git export of $REV"
"$PYTHON" "$HERE/arms.py" prepare --repo "$ROOT" --dst "$SCRATCH" --rev "$REV"

# --- questions ----------------------------------------------------------------

mapfile -t QIDS < <("$PYTHON" - "$HERE/questions.yaml" "$IDS" <<'PY'
import sys, yaml
ids = [s for s in sys.argv[2].split(",") if s] if len(sys.argv) > 2 else []
qs = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for q in qs:
    if not ids or q["id"] in ids:
        print(q["id"])
PY
)
[[ ${#QIDS[@]} -gt 0 ]] || { echo "no questions selected" >&2; exit 1; }
say "${#QIDS[@]} questions: ${QIDS[*]}"

question_text() {
  "$PYTHON" - "$HERE/questions.yaml" "$1" <<'PY'
import sys, yaml
qs = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(next(q["question"] for q in qs if q["id"] == sys.argv[2]))
PY
}

# --- one cold session ----------------------------------------------------------

run_one() {  # arm id
  local arm="$1" id="$2"
  local dir="$SCRATCH/$arm/agent"
  local out="$SCRATCH/transcripts/$arm/$id.ndjson"
  local t0 t1 attempt ok=0
  mkdir -p "$SCRATCH/transcripts/$arm" "$SCRATCH/meta/$arm"
  for attempt in $(seq 1 $((RETRIES + 1))); do
    t0=$(date +%s)
    if (cd "$dir" && timeout "$TIMEOUT_S" "$CLAUDE_BIN" -p "$(question_text "$id")" \
        --output-format stream-json --verbose \
        --permission-mode bypassPermissions --no-session-persistence \
        >"$out.tmp" 2>"$out.err"); then :; fi
    t1=$(date +%s)
    # success = a result event that is not an error
    if tail -c 200000 "$out.tmp" | grep -q '"type":"result"' \
       && ! tail -c 200000 "$out.tmp" | grep -q '"is_error":true'; then
      ok=1
      break
    fi
    say "  [$arm/$id] attempt $attempt failed, $((t1 - t0))s"
  done
  mv "$out.tmp" "$out"
  printf '{"arm": "%s", "id": "%s", "duration_s": %d, "retries": %d, "ok": %d, "ts": "%s"}\n' \
    "$arm" "$id" "$((t1 - t0))" "$((attempt - 1))" "$ok" "$(date -Iseconds)" \
    > "$SCRATCH/meta/$arm/$id.json"
}

if [[ $DRY_RUN -eq 1 ]]; then
  say "dry run — arms built, would run ${#QIDS[@]} sessions per arm from:"
  for a in ${ARMS//,/ }; do echo "  $SCRATCH/$a/agent"; done
  exit 0
fi

# --- run (bounded job pool) -----------------------------------------------------

run_arm() {
  local arm="$1"
  say "=== arm: $arm — cold session per question ==="
  local running=0
  for id in "${QIDS[@]}"; do
    if [[ -s "$SCRATCH/transcripts/$arm/$id.ndjson" ]] \
       && grep -q '"is_error":false' "$SCRATCH/transcripts/$arm/$id.ndjson"; then
      say "  [$arm/$id] already done — skipping (delete to rerun)"
      continue
    fi
    while [[ $running -ge $PARALLEL ]]; do
      wait -n || true
      running=$((running - 1))
    done
    run_one "$arm" "$id" & running=$((running + 1))
  done
  wait || true
}

IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
  run_arm "$arm"
done

# --- score ---------------------------------------------------------------------

say "scoring (grader against the real corpus; judge blind to arms)"
SCORE_ARGS=( "$SCRATCH" --questions "$HERE/questions.yaml" \
  --corpus "$HW/md" --out "$RESULTS" )
if [[ $NO_JUDGE -eq 1 ]]; then SCORE_ARGS+=(--no-judge); fi
"$PYTHON" "$HERE/score.py" "${SCORE_ARGS[@]}"

say "auth sanity: check summary.json — every session erroring with"
say "'Not logged in' means the credentials went stale mid-run: /login, rerun."
say "results: $RESULTS  (delta.md, scoreboard-*.md, transcripts/, answers/)"
