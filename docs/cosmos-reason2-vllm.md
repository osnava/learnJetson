# Runbook: Cosmos-Reason2-2B with vLLM (physical-reasoning VLM, driven from the PC)

> **Verified end-to-end on this board:** 2026-09-01 — all three §4 gates passed (image gate
> **13.8 s** at 150 tokens); serving at 1024/0.55 after the §2.4 preprocessor patch; full
> results in the [completion report](#9-agent-completion-report). The one human gate in the
> flow is the NGC account + API key (§2).
>
> A *deployment* runbook (agent-protocol steps to reach a running stack), not a symptom
> runbook — recovery guides follow the
> [troubleshooting anatomy](troubleshooting/README.md) instead.

## What & why

Serve NVIDIA's **Cosmos-Reason2-2B** — the **FP8 checkpoint from NGC**, not the HF release —
with vLLM on the Jetson, as an **OpenAI-compatible endpoint on port 8010**, driven from the PC
by **Live VLM WebUI** using the PC's webcam.

Cosmos-Reason2 is post-trained for **physical reasoning** (hazards, collisions, spatial state),
which complements this repo's existing VLM path: VILA/nano_llm (README §3) is the lighter
chat/captioning stack. Inference stays fully local; the LAN link is permanent because the
webcam and UI live on the PC — every process kept off the Jetson is memory the model gets.

**Topology:**

```
PC webcam ──► Live VLM WebUI on the PC (https://localhost:8090)
                    │  OpenAI-compatible HTTP
                    ▼
          Jetson: vLLM container (http://<jetson-ip>:8010/v1)
                    │  volume mounts, read-only
                    ├── /ssd/models/cosmos-reason2-2b_v1208-fp8-static-kv8  (FP8 weights, 3.3 GB)
                    └── ~/.cache/vllm  (torch.compile cache — first run builds, later runs reuse)
```

## How to execute this runbook (agent protocol)

Every command block is labeled with the machine it runs on, and every step with a marker:

| Marker | Meaning |
|---|---|
| `[AUTO]` | Agent executes without asking |
| `[CONFIRM]` | Agent prepares, states the action, gets the owner's OK, executes |
| `[HUMAN]` | Only a person at a keyboard/browser can do it — agent waits, provides exact steps |
| `[BLOCKING]` | Nothing downstream may start until this step is done |
| `[GATE]` | Hard-stop verification. On failure: stop, diagnose via [§8](#8-troubleshooting), fix, re-run. Never continue past a red gate |

Actors: **`HOST: jetson`** = over SSH (`agent/jetson.sh ssh <ip> ...`); **`HOST: pc`** = on the
Windows PC that drives the WebUI.

**Failure policy:** any non-zero exit, garbled output, or OOM aborts the current step. Re-run
the failed step once after its documented fix; if it fails again, stop and report — do not
improvise new flag combinations (memory flags are a balanced set; see §4).

**SSH lessons baked in** (from [FIELD_NOTES](../agent/FIELD_NOTES.md) #17, learned the hard way):

- **Env file over exports** — per-machine values live in `~/.cosmos-env` on the Jetson
  (template: [`agent/cosmos-env.example`](../agent/cosmos-env.example)); exports die with the
  SSH session.
- **Scripts over quoted strings** — non-trivial command lines are committed as
  [`agent/launch_vllm.sh`](../agent/launch_vllm.sh), streamed to the Jetson, and run by path.
  Never inline a multi-flag docker command through the Windows→SSH shell layers.
- **`sudo -n` check first** — this box has scoped passwordless sudo only (issue #15: docker,
  journalctl, systemctl for docker/gdm). `sudo -n true` failing is *expected*; plain `docker`
  works (user in the docker group). Anything else sudo (swapoff, ufw) needs the owner →
  `[CONFIRM]`/`[HUMAN]`.
- **`nohup` for long downloads** — multi-GB pulls run detached via `setsid nohup … &` with a
  file-based log, polled from a separate SSH session.

## 0. Machine fit — verify, don't redo

The draft this runbook adapts assumed a stock board. This machine already has everything the
draft's §3.1 would set up. Verified over SSH 2026-09-01:

| Requirement | This machine (verified) | Check command |
|---|---|---|
| JetPack ≥ 6.2.2 / L4T ≥ 36.5.0 (draft), ≥ r36.4 (container floor) | L4T **R36.5.2** (JetPack 6.2.2) | `cat /etc/nv_tegra_release` |
| Headless boot | `multi-user.target` default | `systemctl get-default` |
| MAXN_SUPER + clocks/fan locked at boot | `jetson_clocks.service` active/enabled, `NV Power Mode: MAXN_SUPER` | `systemctl is-active jetson_clocks.service; nvpmodel -q` |
| Free-memory floor (issue #17) | `vm.min_free_kbytes = 131072` (128 MB) | `cat /proc/sys/vm/min_free_kbytes` |
| Page-cache policy | event-time only, before multi-GB loads — `agent/jetson.sh dropcache` | [README §Performance](../README.md#memory-cache-clear-it-for-builds-and-big-loads--not-for-every-run) |

Do **not** *set* clocks/mode manually (`nvpmodel -m …`, `jetson_clocks …`) — the boot service
owns them ([clocks/fan runbook](troubleshooting/lock-clocks-fan-maxn-super.md)). Read-only
queries (`nvpmodel -q`, `systemctl is-active`) are the passwordless verification path on this
box; `sudo jetson_clocks --show` needs the owner's sudo (scoped sudoers, issue #15).

## 1. Storage prep — weights and image on NVMe `[AUTO]`

`[GATE]` ~26 GB needed on `/ssd` (weights ~3.3 GB + vLLM image ~22.3 GB — both measured on
this box 2026-09-01; upstream's ~5 GB / ~8 GB figures understate the unpacked image).
Verified free 2026-09-01: **602 GB**. Ties into the #12 prune policy — images on disk cost
nothing at runtime, running ones do.

```bash
# HOST: jetson  [AUTO]
df -h /ssd                       # need ≥ 13 GB free
mkdir -p /ssd/models ~/.cache/vllm ~/logs
```

## 2. NGC CLI + FP8 weights

The FP8 checkpoint comes from NGC via the NGC CLI (not `huggingface-cli`).

**2.1 Install the CLI** `[AUTO]` — no sudo needed, it unzips under `$HOME`:

```bash
# HOST: jetson  [AUTO]
cd ~ && wget -O ngccli_arm64.zip \
  https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.13.0/files/ngccli_arm64.zip
unzip -o ngccli_arm64.zip && chmod u+x ~/ngc-cli/ngc
export PATH="$PATH:$HOME/ngc-cli"     # for this session; use ~/ngc-cli/ngc by path elsewhere
```

**2.2 NGC account + API key + `ngc config set`** `[HUMAN]` `[BLOCKING]` — irreducible:

1. Sign up / log in at <https://ngc.nvidia.com/> (free; needs `nim` org access).
2. Generate an API key: <https://org.ngc.nvidia.com/setup/api-key>.
3. On the Jetson: `~/ngc-cli/ngc config set` (interactive: paste key, output format `ascii`, org = default).

The key never transits an agent chat or this repo (FIELD_NOTES #19). Nothing downstream starts
until `ngc config set` has succeeded.

**2.3 Download the weights to `/ssd`** `[AUTO]` — per repo storage convention the destination
is `/ssd/models` (the upstream default `~/.cache/huggingface/hub` predates this repo's `/ssd`
layout). ~5 GB over Wi-Fi → nohup pattern:

```bash
# HOST: jetson  [AUTO]
setsid nohup ~/ngc-cli/ngc registry model download-version \
  "nim/nvidia/cosmos-reason2-2b:1208-fp8-static-kv8" \
  --dest /ssd/models > ~/logs/ngc-cosmos.log 2>&1 &

# poll from any session until it exits; auth errors here mean step 2.2 didn't complete:
tail -n 5 ~/logs/ngc-cosmos.log
ls /ssd/models/cosmos-reason2-2b_v1208-fp8-static-kv8   # [GATE] config.json + safetensors present
```

Measured 2026-09-01: **3.3 GB**, 11 files.

**2.4 Shrink the checkpoint's image resolution — REQUIRED on 8 GB** `[AUTO]` — field result
2026-09-01: without this patch vLLM's memory-profiling pass dies at *any* serve config (the
§8 NVML assert); with it, the known-good config serves (§4.2):

```bash
# HOST: jetson  [AUTO]
cd /ssd/models/cosmos-reason2-2b_v1208-fp8-static-kv8
cp preprocessor_config.json preprocessor_config.json.bak
python3 -c "
import json
p = 'preprocessor_config.json'
c = json.load(open(p))
c['size']['longest_edge'] = 50176
c['size']['shortest_edge'] = 3136
json.dump(c, open(p, 'w'), indent=2)
print('patched:', json.load(open(p))['size'])
"
```

Why: the stock config allows 16.7M-pixel images (`longest_edge: 16777216`), and vLLM sizes
its profiling worst case from that — the activation spike kills the engine *regardless of*
`--max-model-len` / `--gpu-memory-utilization`. 50176/3136 is the tutorial's setting for
this board class; the `.bak` restores stock behavior.

## 3. vLLM container — pinned, not floating `[AUTO]`

**Use the exact tag `0.14.0-r36.4-tegra-aarch64-cu126-22.04`.** Reason: newer vLLM builds in
this container family (0.16.0, and the floating `r36.4-tegra-aarch64-cu126-22.04` tag, which
currently serves 0.16.0) produce **gibberish** for Cosmos-Reason2 on Orin — confirmed by two
independent field reports (see [Sources](#sources)); 0.14.0 is the known-good pin. The
`latest-jetson-orin` alias currently resolves to a 0.19.0-era build — untested here; don't.

```bash
# HOST: jetson  [AUTO] — ~22 GB pull (unpacked, measured 2026-09-01), nohup pattern
VLLM_TAG=0.14.0-r36.4-tegra-aarch64-cu126-22.04
setsid nohup docker pull ghcr.io/nvidia-ai-iot/vllm:${VLLM_TAG} > ~/logs/vllm-pull.log 2>&1 &
tail -n 5 ~/logs/vllm-pull.log
```

Note: the upstream llama.cpp fallback for this model listens on 8080, which this box reserves
for open-webui — and open-webui must be stopped for GPU/memory exclusivity anyway (§4), so the
collision is moot but worth remembering. Port map on this box: 5000/5001 Flask, 8080
open-webui, 9001 Roboflow, 8554 WebRTC, 11434 Ollama — **8010 free** (verified: nothing
listening, 2026-09-01).

## 4. Serve + verification gates

**4.0 GPU/memory exclusivity** `[GATE]` `[AUTO]` — one heavy service at a time (the 8 GB heap
is the resource; [README service memory policy](../README.md#service-memory-policy--one-heavy-service-at-a-time-inference-mode)).
`docker ps` must show nothing heavyweight (each ultralytics container holds ~2–3 GB resident;
ollama/inference-server likewise) and `free -h` ~7 GB available. Verified idle 2026-09-01:
zero containers, 6.3 GiB available (7.3 GiB after the event-time cache drop the launcher does).
`launch_vllm.sh` enforces this gate and refuses to launch otherwise.

**4.1 Env file + launcher script** `[AUTO]`:

```bash
# HOST: pc  [AUTO] — deliver script + template once (heredoc pattern, hardened:
# stdin-stream instead of a quoted heredoc through the Windows→SSH layers)
scp agent/launch_vllm.sh agent/cosmos-env.example <user>@<jetson-ip>:~/

# HOST: jetson  [AUTO]
cp ~/cosmos-env.example ~/.cosmos-env && nano ~/.cosmos-env   # real values; gitignored by design
chmod +x ~/launch_vllm.sh
```

`~/.cosmos-env` holds `MODEL_PATH`, `VLLM_TAG`, `VLLM_PORT`, and the memory flags. The real
file lives only on the Jetson (never committed — same pattern as
[`agent/inventory.md`](../agent/inventory.md)); the committed template is the documentation.

**4.2 Known-good serve config** — what `launch_vllm.sh` runs (issue #18, confirmed for 8 GB
Orin by the field reports in [Sources](#sources)). The flags live canonically in
[`agent/launch_vllm.sh`](../agent/launch_vllm.sh) + `~/.cosmos-env`; this listing is the
reference copy — change one, change both:

```bash
vllm serve /models/cosmos-reason2-2b --served-model-name cosmos-reason2-2b \
  --host 0.0.0.0 --port 8010 --enforce-eager --max-model-len 1024 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.55 --max-num-seqs 1 \
  --enable-chunked-prefill --limit-mm-per-prompt '{"image":1}' \
  --enable-prefix-caching --reasoning-parser qwen3
```

- **Fallback** (first OOM): `--max-model-len 768` / `--max-num-batched-tokens 768` /
  `--gpu-memory-utilization 0.52`.
- **Never go above 0.60** on this board. Headless freed ~0.8–1 GB — if you raise anything,
  raise **one flag at a time** and re-run the §4 image gate each time.
- **Field-verified ladder (2026-09-01):** without the §2.4 preprocessor patch, *both*
  1024/0.55 and 768/0.52 died in vLLM's profiling pass (the §8 NVML assert). With the patch,
  768/0.52 produced a clean `0.08 GiB KV cache needed > 0.0 GiB available` error, and
  **1024/0.55 serves**: weights 2.91 GiB, KV cache 0.2 GiB (1,872 tokens). The preprocessor
  patch is the prerequisite, not a last resort.

**4.3 Launch + gates** `[AUTO]` — `./launch_vllm.sh` does preflight → serve → readiness poll,
then run the three gates by hand:

```bash
# HOST: jetson  [AUTO]
./launch_vllm.sh            # exit 0 = ready · 2 preflight · 3 container died · 4 timeout
./launch_vllm.sh logs       # follow vLLM logs while gates run
```

**`[GATE]` 1 — model listed:**

```bash
curl -s http://localhost:8010/v1/models        # must list "cosmos-reason2-2b"
```

**`[GATE]` 2 — text smoke test returns coherent prose:**

```bash
curl -s http://localhost:8010/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "cosmos-reason2-2b",
  "messages": [{"role": "user", "content": "What capabilities do you have?"}],
  "max_tokens": 128
}' | python3 -m json.tool
```

Garbled tokens here = container version bug → §8, *not* a memory problem.

**`[GATE]` 3 — image described correctly, wall-clock latency recorded** (this number sets the
WebUI Frame Interval in §5). Test image: the classic ultralytics `bus.jpg` — a street scene
with a bus and people, rich in spatial relationships. The payload is built by a script, not an
inline JSON string (SSH quoting lesson) — and it embeds the image as base64, because vLLM
blocks `file://` media URLs unless `--allowed-local-media-path` is served with:

```bash
# HOST: jetson  [AUTO]
wget -q -O /tmp/bus.jpg \
  https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg
python3 <<'EOF'
import base64, json
img = base64.b64encode(open("/tmp/bus.jpg", "rb").read()).decode()
payload = {"model": "cosmos-reason2-2b", "max_tokens": 150, "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}},
    {"type": "text", "text": "Describe the scene. Where are the people relative to the bus, and is anyone at risk of being hit if it moves?"}]}]}
json.dump(payload, open("/tmp/gate3.json", "w"))
print("payload written:", len(img), "b64 chars")
EOF
time curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" -d @/tmp/gate3.json | python3 -m json.tool
```

Pass = correct scene description (bus, people, positions) + a latency figure for the report.

## 5. PC side — Live VLM WebUI `[HUMAN]` hands off from here

**5.0 PC → Jetson reachability (the firewall decision, executed)** `[AUTO]` → `[CONFIRM]`:

```bash
# HOST: pc  [AUTO]
curl -s http://<jetson-ip>:8010/v1/models | grep -q cosmos-reason2-2b && echo "reachable"
```

If that fails while the §4 gates passed on the Jetson itself, check the firewall
(`sudo ufw status` on the Jetson — needs the owner's sudo, so `[CONFIRM]`): **only if** it
shows active, ask the owner before `sudo ufw allow 8010/tcp` (opening ports is
ask-human-first per [`agent/AGENTS.md`](../agent/AGENTS.md)). If ufw is inactive — the likely
case on this box, whose LAN services (5000/8080/9001) have never needed port work — the
failure is elsewhere: re-check the URL uses the Jetson's *current* DHCP IP (`agent/jetson.sh
find` / `agent/inventory.md`) and plain `http://`.

**5.1 Install** `[AUTO]` on the PC (any user account with Python 3.10+; on Windows use WSL2 —
the project's supported path, and the browser-side webcam is unaffected since the *browser*
captures the frames — field setup 2026-09-01: WSL Ubuntu-22.04 + uv):

```bash
# HOST: pc  [AUTO]
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
uv venv ~/.live-vlm --python 3.12 && source ~/.live-vlm/bin/activate
uv pip install live-vlm-webui
live-vlm-webui                       # serves https://localhost:8090
```

(Docker alternative: `git clone https://github.com/NVIDIA-AI-IOT/live-vlm-webui && cd
live-vlm-webui && ./scripts/start_container.sh` — same port.)

**5.2 Browser steps** `[HUMAN]` — irreducible (cert + camera are browser-permission walls):

1. Open <https://localhost:8090> → accept the self-signed certificate (**Advanced → Proceed**).
2. Grant camera permission when prompted.
3. **VLM API Configuration** (left sidebar): API Base URL
   `http://<jetson-ip>:8010/v1` — note `http`, not `https`, and the Jetson's *LAN* IP, not
   `localhost` (vLLM and the WebUI are on different machines). Click **Refresh**, pick
   `cosmos-reason2-2b`.
4. Set the tuning values below **before** clicking Start.

**5.3 Tuning on 8 GB** — the context window is 1024 tokens and one request runs at a time:

| Setting | Value | Why |
|---|---|---|
| Max Tokens | **100–150** | image tokens consume ~500–600 of the 1024 window; defaults (512) 400-error |
| Frame Interval | **450+ (measured 2026-09-01)** | at 30 fps, interval ≥ 30 × gate-3 seconds: 13.8 s × 30 ≈ 415, rounded up so requests never queue |
| Prompt shape | structured physical-reasoning prompts | see below |

**Prompt shape:** Cosmos-Reason2 was post-trained for physical reasoning — ask physics-of-the-
scene questions, not open-ended captioning. Good: *"List objects at risk of falling, any
collision hazards, and where things are relative to each other. Be concise."* Wasteful:
*"Tell me everything you see."* (burns the token window the reasoning needs).

## 6. Restart cheat sheet (this machine)

Clocks/fan/MAXN_SUPER auto-lock at every boot — nothing to do there. After a reboot or a
session running something else, serving again is three steps:

```bash
# HOST: jetson
docker ps                                     # anything running? stop it (exclusivity gate)
~/launch_vllm.sh                              # dropcache + serve + readiness poll are inside
# HOST: pc
live-vlm-webui                                # or restart its container; browser tab stays valid
```

If latency turns erratic with swap in play, add `sudo swapoff -a` before launch as a
launch-time-only step (see §7) — otherwise leave swap alone.

## 7. Decisions made for this machine

| Decision | Choice | Reason |
|---|---|---|
| **Swap policy** | First serve attempt with swap **untouched** (16 GB `/ssd` swapfile + zram stay on) | The memory floor (#17) + headless already give headroom; swap is the safety net, not extra speed. If gate-3 latency is erratic (swap-in visible in `free -m`), make `sudo swapoff -a` a launch-time-only `[CONFIRM]` cheat-sheet step (needs the owner's sudo). **Never disable zram permanently.** Outcome (2026-09-01): swap untouched, latency steady — no swapoff step warranted. |
| **Port 8010** | Serve on 8010 | Free on this box (verified 2026-09-01). The draft's llama.cpp fallback port 8080 collides with open-webui — moot because open-webui must be stopped anyway for exclusivity, but 8010 avoids the question entirely. |
| **Firewall** | Check `sudo ufw status` at deploy time; open 8010/tcp **only if active** | Could not be verified non-interactively (`ufw status` needs the owner's sudo). LAN services (5000/8080/9001) have always been reachable without port work here, suggesting ufw is inactive. Opening ports is ask-human-first per [`agent/AGENTS.md`](../agent/AGENTS.md). |
| **GPU exclusivity as a gate** | Hard `[GATE]` inside `launch_vllm.sh`: no other containers + ~7 GB available | 8 GB unified heap; ultralytics ~2–3 GB resident each, ollama similar. Nothing else fits alongside — no ROS 2 / second stack on this board while serving. |

## 8. Troubleshooting

| Symptom | Fix | Not this |
|---|---|---|
| **Engine dies at startup: `NVML_SUCCESS == r INTERNAL ASSERT FAILED` (CUDACachingAllocator) during init** | Apply the §2.4 preprocessor patch — the 16.7M-pixel default makes vLLM's profiling pass OOM; the assert is Jetson's `NvMapMemAlloc` ENOMEM surfacing through PyTorch. Field-verified 2026-09-01: both primary and fallback configs died identically without it; 1024/0.55 serves with it. | Not memory-flag tuning — the profiling workload is driven by image resolution, not `--max-model-len` |
| **Gibberish output** (repeated tokens, mixed scripts) | Pin the container to vLLM **0.14.0** (`0.14.0-r36.4-tegra-aarch64-cu126-22.04`) before touching anything else. 0.16.0-era builds show this on Cosmos-Reason2; it mimics a tokenizer bug. | Do **not** tune memory flags in response — it's a container-version bug. |
| **Text passes, images OOM / context overflow** | Raise `--max-model-len` (768 → 1024) | Not `--gpu-memory-utilization` — image tokens need context space, not KV pool fraction; and never above 0.60 on this board. |
| **`400 max_tokens is too large` from WebUI** | WebUI Max Tokens down to 100–150 (image tokens eat ~500–600 of the window) | — |
| **Model missing in WebUI dropdown** | Check `curl http://<jetson-ip>:8010/v1/models` from the PC; API Base URL must be `http://` + Jetson LAN IP, not `https`/`localhost` | — |
| **vLLM won't start: OOM at load** | Fallback config (§4.2); verify exclusivity gate actually passed (`docker ps` empty) | Don't stack changes — one flag at a time, re-run gates. |
| **Slow inference** | Expected: this config prioritizes fitting in 8 GB over speed. Shorten Max Tokens, raise Frame Interval | — |
| **Anything else heavy needed alongside** | It doesn't fit — 8 GB leaves no headroom for ROS 2 or a second stack while serving (this is a documented platform limit, not a tuning failure) | — |

## 9. Agent completion report

Fill on first successful end-to-end run (this is the issue #18 acceptance record). Filled 2026-09-01:

- [x] JetPack / L4T version serving ran on: **JetPack 6.2.2 / L4T R36.5.2**
- [x] Container tag used + pinned?: **`0.14.0-r36.4-tegra-aarch64-cu126-22.04` — pinned**
- [x] Gate 1 `/v1/models` result: **lists `cosmos-reason2-2b`, `max_model_len: 1024`**
- [x] Gate 2 text output coherent?: **yes — coherent prose, none of the 0.16.0-style gibberish**
- [x] Gate 3 image description correct? + wall-clock latency: **yes (bus scene, six people, positions/proximity correct) — 13.8 s**
- [x] WebUI Frame Interval set from that latency: **450+ (30 fps × 13.8 s ≈ 415)**
- [x] Final `--max-model-len` / `--gpu-memory-utilization`: **1024 / 0.55** — weights 2.91 GiB, KV cache 0.2 GiB (1,872 tokens)
- [x] Swap decision outcome: **untouched** — 16 GB `/ssd` swapfile + zram stayed on, latency steady
- [x] Deviations: **§2.4 preprocessor patch is required, not optional; brief fallback detour (768/0.52) taken while diagnosing; measured sizes differ from upstream (weights 3.3 GB vs ~5 GB, image 22.3 GB vs ~8 GB)**

## Sources

- [Cosmos Reason 2 2B — Jetson AI Lab model page](https://www.jetson-ai-lab.com/models/cosmos-reason2-2b/) — NGC CLI steps, Orin-Nano constrained flag set, prefix-caching + `--reasoning-parser qwen3`
- [Cosmos Reason2 on Jetson — Jetson AI Lab tutorial](https://www.jetson-ai-lab.com/tutorials/cosmos-reason2-vlm/) — full flow this runbook adapts; WebUI settings; `preprocessor_config.json` lever
- [Deploying Open Source VLMs on Jetson — NVIDIA on HF](https://huggingface.co/blog/nvidia/cosmos-on-jetson) — canonical deployment writeup; comments hold the 8 GB field data: `--max-model-len 1024` / `--gpu-memory-utilization 0.55` for vision (Lawrence-okolo) and the 0.16.0→0.14.0 gibberish fix (tharindupr)
- [Cosmos-Reason2-2B model card](https://huggingface.co/nvidia/Cosmos-Reason2-2B) · [NGC FP8 checkpoint](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/cosmos-reason2-2b)
- [Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) — install paths, :8090, settings semantics
- [vLLM container tags](https://github.com/orgs/nvidia-ai-iot/packages/container/package/vllm) — tag list as observed 2026-09-01 via the ghcr.io registry API (0.14.0 / 0.16.0 / 0.19.0-era r36.4 tegra tags)
- Machine-fit values: live SSH verification 2026-09-01 (this repo, `agent/jetson.sh`)
