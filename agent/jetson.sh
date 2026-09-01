#!/usr/bin/env bash
# jetson.sh — AI-agent-friendly control panel for the Jetson Orin Nano.
# Companion to agent/AGENTS.md (read it first — especially "Forbidden").
#
# Usage:
#   ./jetson.sh find [subnet]        discover the Jetson on the LAN (+ USB-C fallback)
#   ./jetson.sh ssh <ip> [cmd...]    open a shell (or run a command) with sane SSH defaults
#   ./jetson.sh status <ip>          full health panel (rootfs, mounts, swap, docker, units)
#   ./jetson.sh health <ip>          silent check; exit 0 = healthy (for agents/CI)
#   ./jetson.sh logs <ip> [unit]     tail journal (default: -b boot log)
#   ./jetson.sh dropcache <ip>       sync + drop page cache (privileged container; before TRT builds / multi-GB loads only)
#
# Tunables via environment or agent/inventory.sh (gitignored):
#   JETSON_USER      SSH user            (default: from inventory or $USER)
#   JETSON_SUBNET    default sweep CIDR-less prefix, e.g. 192.168.1  (default: 192.168.1)
#   JETSON_ROOTFS    expected root device for health   (default: /dev/nvme0n1p1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[ -f "$SCRIPT_DIR/inventory.sh" ] && . "$SCRIPT_DIR/inventory.sh"
JETSON_USER="${JETSON_USER:-$USER}"
JETSON_SUBNET="${JETSON_SUBNET:-192.168.1}"
JETSON_ROOTFS="${JETSON_ROOTFS:-/dev/nvme0n1p1}"
RNDIS_IP="192.168.55.1"   # USB-C device-mode fallback — always this address when the OS is up

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new)

port_open() { timeout 1 bash -c "exec 3<>/dev/tcp/$1/22" 2>/dev/null; }

cmd_find() {
  local subnet="${1:-$JETSON_SUBNET}"
  echo "Sweeping ${subnet}.0/24 for SSH (port 22)..."
  local found=()
  while read -r ip; do found+=("$ip"); done < <(
    seq 1 254 | sed "s/^/$subnet./" | xargs -P 32 -I{} bash -c \
      'timeout 1 bash -c "exec 3<>/dev/tcp/{}/22" 2>/dev/null && echo {}' 2>/dev/null || true
  )
  if [ ${#found[@]} -gt 0 ]; then
    printf 'LAN candidates: %s\n' "${found[*]}"
  else
    echo "No SSH found on LAN."
  fi
  if port_open "$RNDIS_IP"; then
    echo "USB-C fallback UP: ssh ${JETSON_USER}@${RNDIS_IP} (RNDIS device mode)"
  else
    echo "USB-C fallback down (cable unplugged, or OS not up — see AGENTS.md discovery)."
  fi
  [ ${#found[@]} -gt 0 ] || port_open "$RNDIS_IP" || {
    echo "Nothing reachable. Machine is off or pre-boot-dead — escalate per the black-screen runbook."
    return 1
  }
}

cmd_ssh() {
  local ip="$1"; shift || true
  ssh "${SSH_OPTS[@]}" "${JETSON_USER}@${ip}" "$@"
}

remote_panel() {
  ssh "${SSH_OPTS[@]}" "${JETSON_USER}@$1" '
    echo "rootfs    : $(findmnt -n -o SOURCE /)  [$(cat /proc/cmdline | tr " " "\n" | grep "^root=" || echo ?)]"
    echo "uptime    : $(uptime -p | cut -d" " -f2-)"
    echo "/ssd      : $(findmnt -n -o SOURCE,SIZE,USED /ssd 2>/dev/null || echo "not mounted")"
    echo "swap      : $(swapon --show --noheadings 2>/dev/null | awk "{print \$1\" (\"\$3\")\"}" | tr "\n" " ")"
    echo "mem       : $(free -m | awk "NR==2{print \$4\"M free / \"\$7\"M avail\"}") / min_free_kbytes $(cat /proc/sys/vm/min_free_kbytes)kB"
    echo "docker    : $(systemctl is-active docker) / failed units: $(systemctl --failed --no-legend | wc -l)"
    echo "boot order: $(efibootmgr 2>/dev/null | grep BootOrder || echo n/a)"
  '
}

cmd_status() { remote_panel "$1"; }

cmd_health() {
  local ip="$1" out
  out="$(remote_panel "$ip")" || { echo "unreachable"; return 2; }
  echo "$out"
  echo "$out" | grep -q "rootfs    : ${JETSON_ROOTFS}" || { echo "FAIL: unexpected rootfs"; return 1; }
  echo "$out" | grep -q "failed units: 0" || { echo "FAIL: failed units present"; return 1; }
  echo "$out" | grep -q "docker    : active" || { echo "FAIL: docker inactive"; return 1; }
  echo "OK"
}

cmd_logs() {
  local ip="$1" unit="${2:-}"
  if [ -n "$unit" ]; then
    cmd_ssh "$ip" "journalctl -u $unit -n 100 --no-pager"
  else
    cmd_ssh "$ip" "journalctl -b -n 100 --no-pager"
  fi
}

cmd_dropcache() {
  local ip="$1"
  echo "Dropping page cache on ${ip} (sync + drop_caches=3 via privileged container)..."
  cmd_ssh "$ip" "docker run --rm --privileged alpine:latest sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'"
  echo "done — per policy: before TRT builds / multi-GB loads only, never daemonized (README §Performance)"
}

case "${1:-help}" in
  find)   shift; cmd_find "$@" ;;
  ssh)    shift; cmd_ssh "$@" ;;
  status) shift; cmd_status "$@" ;;
  health) shift; cmd_health "$@" ;;
  logs)   shift; cmd_logs "$@" ;;
  dropcache) shift; cmd_dropcache "$@" ;;
  *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
