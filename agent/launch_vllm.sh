#!/usr/bin/env bash
# launch_vllm.sh — serve Cosmos-Reason2-2B (FP8, NGC checkpoint) with vLLM on a
# Jetson Orin Nano 8 GB. Companion to docs/cosmos-reason2-vllm.md (§4); read it
# before changing any serve flag — they are a memory-balanced set, not defaults.
#
# Runs ON THE JETSON. Deploy it from the repo (heredoc pattern, hardened:
# stdin-stream beats a quoted heredoc through the Windows->SSH layers):
#   ssh <user>@<jetson-ip> 'cat > ~/launch_vllm.sh && chmod +x ~/launch_vllm.sh' < agent/launch_vllm.sh
# Per-machine values live in ~/.cosmos-env (template: agent/cosmos-env.example) —
# env file over exports, so values survive every new SSH session.
#
# Usage:
#   ./launch_vllm.sh          preflight gates -> serve -> readiness poll
#   ./launch_vllm.sh logs     follow the vLLM container log
#   ./launch_vllm.sh stop     stop and remove the container
# Exit codes: 0 ready · 2 preflight gate failed · 3 container died while polling · 4 readiness timeout
set -euo pipefail

ENV_FILE="${COSMOS_ENV:-$HOME/.cosmos-env}"
[ -f "$ENV_FILE" ] || { echo "FAIL: $ENV_FILE not found — copy agent/cosmos-env.example to ~/.cosmos-env and fill it in (runbook §4)."; exit 2; }
# shellcheck disable=SC1090
. "$ENV_FILE"

: "${MODEL_PATH:?MODEL_PATH must point at the NGC checkpoint dir (e.g. /ssd/models/cosmos-reason2-2b_v1208-fp8-static-kv8)}"
: "${VLLM_TAG:=0.14.0-r36.4-tegra-aarch64-cu126-22.04}"  # pinned: 0.16.0-era builds garble Cosmos output (runbook §8)
: "${VLLM_PORT:=8010}"
: "${VLLM_MAX_MODEL_LEN:=1024}"        # known-good on 8 GB; fallback 768 (runbook §4)
: "${VLLM_MAX_BATCHED_TOKENS:=$VLLM_MAX_MODEL_LEN}"
: "${VLLM_GPU_MEM_UTIL:=0.55}"         # known-good; fallback 0.52; NEVER above 0.60
: "${VLLM_CONTAINER:=cosmos-vllm}"
: "${VLLM_CACHE_DIR:-$HOME/.cache/vllm}"
: "${READY_TIMEOUT_SECS:=900}"         # first launch compiles kernels; later runs reuse the cache
IMAGE="ghcr.io/nvidia-ai-iot/vllm:${VLLM_TAG}"
API="http://localhost:${VLLM_PORT}"

fail_gate() { echo "GATE FAILED: $*"; exit 2; }

dump_logs() { docker logs --tail 40 "$VLLM_CONTAINER" 2>&1 || true; }

preflight() {
  echo "== preflight gates (runbook §4) =="
  command -v curl >/dev/null || fail_gate "curl not installed (sudo apt install curl)"
  command -v docker >/dev/null || fail_gate "docker not available"
  # Self-remove any previous instance first, so the memory gate below measures
  # what serving will actually have (the old container still holds its memory).
  docker rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 || true
  # [GATE] GPU/memory exclusivity — one heavy service at a time (README service
  # memory policy). Our own container is gone (above); alpine (agent ops, 8 MB)
  # is the policy's one "always fine" exception — everything else blocks.
  local others
  others="$(docker ps --format '{{.Names}} {{.Image}}' \
    | awk -v me="$VLLM_CONTAINER" '$1 == me {next} $2 ~ /^alpine(:|$)/ {next} {print $1}' || true)"
  [ -z "$others" ] || fail_gate "other containers running ($others) — stop them first; nothing heavyweight may share this 8 GB heap"
  # Event-time cache drop: a multi-GB model load is exactly the policy's trigger
  echo "dropping page cache (event-time: multi-GB model load)"
  docker run --rm --privileged alpine:latest sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' >/dev/null
  local avail_mb
  avail_mb="$(free -m | awk 'NR==2{print $7}')"
  [ "$avail_mb" -ge 6500 ] || fail_gate "only ${avail_mb} MB memory available (want ~7 GB) — stop heavyweight services, then retry"
  [ -d "$MODEL_PATH" ] || fail_gate "MODEL_PATH '$MODEL_PATH' does not exist — run the NGC download first (runbook §2)"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image $IMAGE not local — pulling (~22 GB; runbook §3 has the nohup pattern)..."
    docker pull "$IMAGE"
  fi
}

serve() {
  echo "== launching $VLLM_CONTAINER ($IMAGE) =="
  mkdir -p "$VLLM_CACHE_DIR"
  docker run -d --name "$VLLM_CONTAINER" \
    --runtime nvidia --network host \
    -v "$MODEL_PATH":/models/cosmos-reason2-2b:ro \
    -v "$VLLM_CACHE_DIR":/root/.cache/vllm \
    "$IMAGE" \
    vllm serve /models/cosmos-reason2-2b \
      --served-model-name cosmos-reason2-2b \
      --host 0.0.0.0 --port "$VLLM_PORT" \
      --enforce-eager \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --max-num-batched-tokens "$VLLM_MAX_BATCHED_TOKENS" \
      --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
      --max-num-seqs 1 \
      --enable-chunked-prefill \
      --limit-mm-per-prompt '{"image":1}' \
      --enable-prefix-caching \
      --reasoning-parser qwen3
}

wait_ready() {
  echo "== readiness poll: GET $API/v1/models (timeout ${READY_TIMEOUT_SECS}s) =="
  local deadline=$(( $(date +%s) + READY_TIMEOUT_SECS ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! docker ps --format '{{.Names}}' | grep -qx "$VLLM_CONTAINER"; then
      echo "FAIL: container exited before ready — last 40 log lines:"
      dump_logs
      exit 3
    fi
    if curl -fsS "$API/v1/models" 2>/dev/null | grep -q cosmos-reason2-2b; then
      echo "READY — $API/v1/models lists cosmos-reason2-2b"
      echo "next: the three verification gates (runbook §4)"
      return 0
    fi
    sleep 5
  done
  echo "TIMEOUT after ${READY_TIMEOUT_SECS}s — last 40 log lines:"
  dump_logs
  exit 4
}

case "${1:-serve}" in
  serve) preflight; serve; wait_ready ;;
  logs)  exec docker logs -f "$VLLM_CONTAINER" ;;
  stop)  docker rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 && echo "stopped + removed $VLLM_CONTAINER" || echo "no $VLLM_CONTAINER container" ;;
  *) echo "usage: launch_vllm.sh [serve|logs|stop]"; exit 2 ;;
esac
