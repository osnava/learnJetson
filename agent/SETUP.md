# SETUP.md — From bare kit to agent-operated Jetson

> For someone with a fresh **Jetson Orin Nano Developer Kit** and this repo.
> The kit in `agent/` (rulebook `AGENTS.md`, control script `jetson.sh`,
> `inventory` template, troubleshooting runbooks) works with **any** AI-agent
> harness that can open an SSH connection — it needs no vendor-specific tooling.
> Lessons behind these steps, with sources, live in
> [FIELD_NOTES.md](FIELD_NOTES.md). Every non-obvious claim here is cited.

## 0. What you end up with

A headless Jetson on your LAN, discoverable by one command, with: storage and
docker on NVMe, clocks+fan locked at boot, a scoped-sudo policy for the agent,
the YOLO/TensorRT stack in containers — and an agent that knows the rules it may
operate under. Each step has a **verify** line; don't move on until it passes.

## 1. Hardware

- Jetson Orin Nano Developer Kit (any revision — flashing JetPack 6.2+ turns
  every Orin Nano dev kit into the "Super" variant: 25 W / MAXN_SUPER modes,
  ~1.7× TOPS — [NVIDIA JetPack 6.2 announcement](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)).
- NVMe SSD (M.2 2230/2280) — strongly recommended; the kit boots from SD but
  everything heavy lives better on NVMe.
- microSD card ≥ 64 GB (the boot disk on this design).
- Active cooling fan (required for 25 W/MAXN_SUPER sustained loads).
- USB-C **data** cable (this is also your emergency console, see step 3).
- Optional: USB-TTL serial adapter (~$5) — Button Header TX→pin 3, RX→pin 4,
  GND→pin 7, 115200 8N1; shows every boot stage headlessly.

## 2. Flash JetPack

Write the JetPack SD image to the microSD card (NVIDIA's SD Card Manager or
`dd`), insert NVMe + SD, boot. Use the latest JetPack 6.x — Super mode and the
higher clocks need 6.2+ (same source as above).

**Verify:** first-boot OLED/HDMI wizard completes; `cat /etc/nv_tegra_release`
shows an R36.x L4T release.

## 3. First access — and the cable that always works

Put the Jetson on your LAN (Wi-Fi or Ethernet). DHCP addresses change per
lease, so discovery beats hardcoding. Critically, **every Orin developer kit
exposes a fixed `192.168.55.1` over USB-C (USB device mode)** — a data cable
is a network link that works whenever the OS is up, no infrastructure needed
([Jetson AI Lab — Getting Started](https://www.jetson-ai-lab.com/tutorials/getting-started-with-jetson/)).
That address is your lifeline whenever Wi-Fi misbehaves.

Set up key-based SSH (never share the account password in a chat with an
agent — see step 8):

```bash
ssh-copy-id <user>@<jetson-ip>
ssh -o BatchMode=yes <user>@<jetson-ip> true && echo key-auth-ok
```

**Verify:** `ssh -o BatchMode=yes` succeeds with no prompt.

## 4. Install the agent kit

Copy `agent/` to the repo you'll operate from, then fill in your machine's
facts:

- `agent/inventory.sh` — SSH user, LAN subnet prefix, expected root device
  (drives `jetson.sh find/health` defaults).
- `agent/inventory.md` — human-readable facts (MACs, PARTUUIDs, boot state).
  Both are **gitignored — real identifiers never go in a public repo.**

Adapt `agent/AGENTS.md`: it encodes the safety contract for any agent on this
machine (allowed / ask-first / forbidden). Keep the structure even if you
change the specifics.

Fetch the hardware-doc corpus so hardware questions get grounded answers
instead of guesses (datasheet, carrier-board spec, pinmux, thermal/design
guides — converted to greppable markdown, gitignored). Do this **on the PC
you operate the agent from** — the Jetson stores and serves none of it:

```bash
pip install pymupdf4llm openpyxl    # converter (falls back to poppler pdftotext)
agent/hw-docs/fetch.sh              # ~9 MB; --full adds the 66 MB SoC TRM + schematics
```

The data sheet itself sits behind NVIDIA's (free) login — `fetch.sh`
prints the one-time manual step when it detects it. The routing table
from question to doc-section lives in
[`agent/hw-docs/INDEX.md`](hw-docs/INDEX.md).

**Verify:**

```bash
./jetson.sh find          # finds the box on your subnet + probes USB-C fallback
./jetson.sh status <ip>   # rootfs, /ssd, swap, docker, failed units
./jetson.sh health <ip>   # silent pass/fail — usable from CI/scripts
```

## 5. Put storage where the wear and speed are

Format/mount the NVMe (e.g. `/ssd`), then move Docker's data-root onto it —
images and containers are multi-GB and have no business on an SD card
([Jetson AI Lab — SSD + Docker](https://www.jetson-ai-lab.com/tutorials/ssd-docker-setup/)):

```bash
sudo systemctl stop docker
sudo rsync -axPS /var/lib/docker/ /ssd/docker/
# /etc/docker/daemon.json → { "data-root": "/ssd/docker" }
sudo systemctl restart docker
```

**Verify:** `docker info | grep "Docker Root"` shows the NVMe path; a test
container runs.

## 6. Lock performance at boot

On an 8 GB shared-memory box you want deterministic numbers: fix clocks and
fan once, in a systemd oneshot, instead of relying on interactive tools.
The official knobs: `sudo nvidia-gw pmode MAXN_SUPER` (or `jetson_clocks`),
`jetson_clocks --fan` sets maximum PWM — mode table and semantics in the
[Jetson Platform Power & Performance guide](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html).
A worked example (service file + verification) is
[the clocks/fan runbook](../docs/troubleshooting/lock-clocks-fan-maxn-super.md).

⚠️ Don't mix managers: jtop's saved fan profiles re-enable `nvfancontrol` and
fight a boot-time lock (see FIELD_NOTES §performance). Pick one owner of the
fan — here it's the systemd service, and jtop is monitor-only.

**Verify:** after reboot, `sudo jetson_clocks --show` lists max clocks; a
10-minute load shows zero clock dips.

## 7. Memory policy

- **Headless if you can** — disabling the desktop frees ~800 MB
  ([Jetson AI Lab — RAM optimization](https://www.jetson-ai-lab.com/tutorials/ram-optimization/)):
  `sudo systemctl set-default multi-user.target`.
- **Swap**: keep zram and add an NVMe swapfile sized ~2× RAM for
  model-loading spikes (commands in the same ai-lab page).
- **Free-memory floor for TensorRT builds**: `vm.min_free_kbytes` =
  128–256 MB via `/etc/sysctl.d/` makes TRT's free-RAM headroom checks pass
  without cache-dropping rituals. The kernel docs define both knobs — and
  their limits (`min_free_kbytes` too high "will OOM your machine instantly";
  `drop_caches` is "not a means to control the growth of the various kernel
  caches" — they are "automatically reclaimed") —
  [kernel.org, /proc/sys/vm](https://docs.kernel.org/admin-guide/sysctl/vm.html).
- The full policy (when `drop_caches` genuinely helps, what never to do) is
  in the [README §Performance optimization](../README.md#performance-optimization)
  and FIELD_NOTES §memory.

**Verify:** `cat /proc/sys/vm/min_free_kbytes` shows your floor after reboot;
`free -m` shows ~6 GB available before loading models.

## 8. Give the agent scoped root — not a password

Two mechanisms, both better than sharing your password with a chat:

1. **docker group** membership: lets the agent read/write root-owned files and
   run one-shot privileged containers (`--privileged` for sysctls) without any
   sudo at all.
2. **Scoped sudoers drop-in** (`/etc/sudoers.d/<name>`, mode 0440) whitelisting
   exactly the guarded surface your AGENTS.md allows, e.g.:

   ```
   <user> ALL=(root) NOPASSWD: /usr/bin/systemctl restart docker, /usr/bin/systemctl start docker, /usr/bin/systemctl stop docker, /usr/bin/journalctl, /usr/bin/docker
   ```

   Validate with `visudo -c` before and after; test that non-whitelisted
   commands still refuse (`sudo -n true` must fail).

If a password was ever shared with an agent session, **rotate it** — and never
route secrets through a chat transcript (generate on the box, move by scp).

**Verify:** `sudo -n systemctl restart docker` works; `sudo -n true` fails.

## 9. The ML stack

Run YOLO from NVIDIA's container rather than pip on the host — it carries the
Jetson-built torch/TensorRT pairing
([Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)):

```bash
docker run -it --runtime nvidia -v /ssd:/ssd ultralytics/ultralytics:latest-jetson-jetpack6
yolo export model=yolo11n.pt format=engine device=0   # TensorRT engine
```

Field-tested gotchas (all in FIELD_NOTES §ml): export from a suffixed copy of
the `.pt` (`yolo export` overwrites the sibling engine), engines are
TensorRT-version/GPU-arch specific, `imgsz` is baked in, and bare
`trtexec`-built engines lack the ultralytics metadata header that task
inference needs.

**Verify:** `yolo predict model=yolo11n.engine source=<img> device=0` returns
detections at ~2–4 ms inference on the locked-clock Orin Nano Super.

## 10. Know the hardware limits before you plan around them

- **No hardware video encoder (no NVENC) on Orin Nano** — decode only (NVDEC).
  `nvv4l2h264enc` and friends cannot work; encode on CPU
  ([NVIDIA forums](https://forums.developer.nvidia.com/t/encode-video/358433),
  [RidgeRun](https://www.ridgerun.com/post/jetson-orin-nano-how-to-achieve-real-time-performance-for-video-encoding)).
- **INT8 TensorRT export of YOLO segmentation models fails** on current
  TensorRT/Jetson stacks ("Could not find any implementation for node …
  Conv + PWN…") — a confirmed TensorRT limitation of the fused seg-head
  patterns, not a fixable config error
  ([ultralytics #21281](https://github.com/ultralytics/ultralytics/issues/21281),
  [#16415](https://github.com/ultralytics/ultralytics/issues/16415));
  detection models export INT8 fine.
- **pip OpenCV has no GStreamer** — `cv2.VideoCapture(..., CAP_GSTREAMER)`
  silently isn't available in pip environments, including the ultralytics
  container
  ([opencv-python #530](https://github.com/opencv/opencv-python/issues/530));
  only a Jetson-built OpenCV (or DeepStream) gets zero-copy NVDEC pipelines.

## Final checklist

| ✓ | Check |
|---|---|
| ☐ | `jetson.sh find` locates the box; USB-C `192.168.55.1` known fallback |
| ☐ | `jetson.sh status` green: NVMe rootfs/data, docker on NVMe, 0 failed units |
| ☐ | Clocks/fan locked at boot; sustained load without dips |
| ☐ | Memory floor active; ~6 GB available at idle |
| ☐ | Agent key-auth works; scoped sudo validated; no password in any chat |
| ☐ | YOLO engine inference verified in-container |
| ☐ | AGENTS.md adapted; inventory filled (and gitignored) |
| ☐ | Hardware corpus fetched — `ls agent/hw-docs/md/` shows the converted docs; INDEX.md routes |
