# FIELD_NOTES.md — Generic lessons from running an AI agent on a Jetson

> Distilled 2026-09-01 from seven issues of ops/ML work on a Jetson Orin Nano
> 8 GB Super (JetPack 6.2.2, TensorRT 10.3) operated remotely by an AI agent.
> Each lesson: what happened → the portable rule → sources. Internal evidence
> links point at this repo's runbooks and benchmark docs; external links are
> authoritative sources. Built for **any** agent harness (ZCode, Claude Code,
> Cursor…) and any Orin-class Jetson.

## Boot & recovery

**1. The boot loader prefers removable media — plan accordingly.**
NVIDIA's UEFI default order puts removable devices (SD/USB) before
non-removable (NVMe/eMMC), and the L4TLauncher looks for
`extlinux.conf`/`BOOTAA64.efi`. Field consequence: a *live* copy of either on
a data disk can hijack boot ("first found wins").
Rule: never leave bootable files on non-boot disks; when booting rootfs-from-
NVMe with an SD bootloader, treat any stray `extlinux.conf` as a live grenade.
Sources: [NVIDIA UEFI guide](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/Bootloader/UEFI.html)
("Removable devices (SD/USB) take precedence…", default order `usb,nvme,emmc,sd,ufs`);
recovery night in [the migration runbook](../docs/troubleshooting/sd-to-nvme-rootfs-migration.md).

**2. Never power-cycle a hung boot "to retry".**
UEFI marks an OS boot chain `Unbootable` after repeated consecutive failures —
observed on-device: three failed boots made the machine look dead (black
screen) while the fix was a UEFI-menu action, not the power button.
Rule: agents have no hands — for anything physical, hand the human *exact
keystrokes* from a runbook and wait. Baking this into the agent rulebook
([AGENTS.md](AGENTS.md) "Forbidden") is what makes "an AI operates my Jetson"
a safe statement.

**3. Split-boot designs couple kernel updates to both disks.**
With SD bootloader + NVMe rootfs, a kernel-package upgrade writes new
`/boot/Image`+initrd to the *rootfs* while the SD's loader may expect the old
ones — the box then boots fine once and dies on the next cold boot.
Rule: after any kernel-package upgrade, sync `/boot` artifacts to the boot
disk before rebooting; write it into the migration runbook's maintenance
section so no future session has to rediscover it.

## Performance & thermals

**4. One owner for clocks and fan — enforced at boot.**
Interactive tools fight each other: jtop's *saved* fan profiles re-enable
`nvfancontrol`, which steals the fan back from a `jetson_clocks`-style lock.
The stable pattern is a systemd oneshot that sets MAXN_SUPER + max fan PWM at
every boot; verification instructions and failure story in
[the clocks/fan runbook](../docs/troubleshooting/lock-clocks-fan-maxn-super.md).
Mode semantics (MAXN vs MAXN_SUPER, default 25 W on Super, `jetson_clocks --fan`):
[NVIDIA Platform Power & Performance](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html).

**5. Benchmark against locked clocks, alternating configurations.**
Every number in this repo's performance docs was taken with clocks locked
(lesson 4) and A/B runs alternated to cancel thermal drift. Without both,
before/after comparisons on a Jetson are noise.

## Memory

**6. `drop_caches` is an event-time tool, never a ritual.**
The kernel docs say it plainly: page cache is "automatically reclaimed by the
kernel when memory is needed elsewhere", drop_caches is "not a means to
control the growth of the various kernel caches", and routine use "can cause
performance problems" ([kernel.org](https://docs.kernel.org/admin-guide/sysctl/vm.html)).
The one real consumer on a Jetson: **TensorRT tactic autotuning makes
free-RAM headroom checks without allocating**, so a fat page cache fails
engine builds with no true OOM. Rule: drop cache right before TRT builds and
multi-GB model loads only; never before small-engine loads, never
daemonized. Full policy + measurements: [README §Performance](../README.md#performance-optimization).

**7. A persistent free-memory floor replaces the ritual.**
`vm.min_free_kbytes` (128–256 MB on an 8 GB board) makes the kernel keep
MemFree above the floor at all times, so headroom checks pass without manual
drops. Kernel warning honored: set too high it "will OOM your machine
instantly" ([kernel.org](https://docs.kernel.org/admin-guide/sysctl/vm.html)).
Field test: 4 GB of stepwise allocation held MemFree at ~350 MB with kswapd
reclaiming cache, vs. the 44 MB kernel default that let free memory sink to
the tactic-starvation zone.

**8. Structural beats procedural.**
Disable the GUI (~800 MB), keep zram + NVMe swap, run only the containers you
need — these remove memory pressure that no sysctl ritual can
([Jetson AI Lab — RAM optimization](https://www.jetson-ai-lab.com/tutorials/ram-optimization/)).
Images *on disk* cost nothing at runtime; images *running* do — so a service
concurrency table (which heavy services may overlap) is worth more than any
prune ([README §Service memory policy](../README.md#service-memory-policy--one-heavy-service-at-a-time-inference-mode)).

## Vision & ML stack

**9. Orin Nano decodes in silicon but encodes on CPU.**
No NVENC on Orin Nano — `nvv4l2h264enc` cannot work; MJPEG/H.264 encode is
software ([NVIDIA forums](https://forums.developer.nvidia.com/t/encode-video/358433),
[RidgeRun](https://www.ridgerun.com/post/jetson-orin-nano-how-to-achieve-real-time-performance-for-video-encoding)).
Design consequence: streaming servers keep `cv2.imencode`/software encoders,
and there is nothing to offload them to.

**10. Measure the premise before building the optimization.**
The NVDEC offload looked obvious — yet software 1080p50 H.264 decode measured
**180 fps** (3.6× realtime, ~11% of the frame budget) while every zero-install
GStreamer bridge (subprocess pipe) cost more than it saved: end-to-end 19.2→
13.3 fps. The blocker: pip OpenCV has **no GStreamer** support
([opencv-python #530](https://github.com/opencv/opencv-python/issues/530)),
so in-process zero-copy needs a Jetson-built OpenCV or DeepStream. Numbers and
retry conditions: [NVDEC notes](../docs/performance/nvdec-decode-notes.md).
Rule: acceptance criteria are measurable — measure before, build, measure
after; be ready to close the issue with "premise disproven, here's the data."

**11. Ultralytics engines: export hygiene is a checklist of its own.**
Field-verified gotchas (full detail in
[the INT8 doc](../docs/performance/int8-tensorrt-engines.md)):
`yolo export` **overwrites the sibling `.engine`** (export from a suffixed
copy of the `.pt`); engines are **TensorRT-version and GPU-arch specific**
(rebuild on-device, ship `.pt`/`.onnx` as canonical artifacts); `imgsz` is
**baked in** at export; engines carry a **metadata header** that bare
`trtexec --saveEngine` builds lack — and ultralytics task inference needs it;
task is guessed from the **filename** first (a seg engine not named `-seg`
needs explicit `task=segment`). Export API reference:
[Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson).

**12. INT8 segmentation export fails on current TensorRT — know it's not you.**
`Error Code 10: Could not find any implementation for node … Conv + PWN…` on
seg models in INT8 is a TensorRT limitation of the fused segmentation-head
patterns, confirmed by the ultralytics maintainer
([#21281](https://github.com/ultralytics/ultralytics/issues/21281),
[#16415](https://github.com/ultralytics/ultralytics/issues/16415)).
Detection models export INT8 fine (+27% throughput at −1.5 mAP here). Rule:
bank the calibration caches, write the 2-command retry for after the next
JetPack/TRT upgrade, and stop burning sessions on "fixing" an upstream gap.

**13. INT8 calibration details that cost hours if unknown.**
Calibration uses the **val split** of `data=` (no `split=` arg exists), batch
1, MinMax; the cache is `<stem>.cache` and is **reused verbatim** on rebuilds;
auto-download pulls ~20 GB of COCO when calibration needs only val2017.
Documented with build logs in [the INT8 doc](../docs/performance/int8-tensorrt-engines.md).

## Containers

**14. The container is the compatibility unit.**
The host stack (torch, TRT python) and the container stack are different
worlds; the ultralytics container is the supported way to run `yolo`
([Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)).
It ships **without flask** — install per throwaway container or bake your own
image. Move docker's data-root to NVMe
([Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/ssd-docker-setup/)),
and remember TRT engines are **not** portable across the container/host TRT
versions either.

**15. One inference loop per service, viewers share it.**
A naive Flask MJPEG server runs a full capture+inference pipeline *per
viewer*: cameras allow one reader, the same TRT engine gets called from N
threads (unsafe), and cost scales with viewers. A single background
capture→infer→encode loop publishing the latest JPEG via a condition variable
gives every viewer flat fps at constant total cost
([README §1](../README.md#1-vision-processing-object-detection),
benchmark in issue #9).

## Agent workflow & git hygiene

**16. Give the agent a rulebook, machine facts, and hands.**
The working kit is three files: `AGENTS.md` (allowed / ask-first / forbidden —
the safety contract), `inventory.{sh,md}` (gitignored per-machine facts), and
`jetson.sh` (find/ssh/status/health/logs/dropcache). Any harness that can SSH
can operate it. Fresh sessions bootstrap in one read; verification after every
change is a single command (`jetson.sh status`).

**17. Shell-layer quoting eats hours — write files, don't inline.**
Non-trivial python/bash through Windows→SSH→docker layers mangles `$` and
quotes silently (a swallowed `$var` wrote to the wrong filename twice).
Rules that held: write scripts locally, `scp` them, run by path; detached
remote jobs via `setsid nohup … &` with **file-based progress logs** polled
from a separate SSH session; `pkill -f` matches your own ssh command line —
use the bracket trick (`pkill -f 'scrip[t].sh'`).

**18. Line endings and exec bits sabotage cross-OS work.**
One `.gitattributes` line — `* text=auto eol=lf` — ends phantom whole-file
diffs between Windows and Linux clones ([gitattributes docs](https://git-scm.com/docs/gitattributes)).
Two silent Windows hazards: NTFS can't represent the executable bit (a
Windows rewrite of a shell script drops mode 100755 — restore with
`git update-index --chmod=+x`), and `autocrlf` can quietly CRLF-inflate
binary-ish working copies. Check `git ls-files --eol` when things smell.

**19. Secrets never transit chats.**
If a credential ever appeared in an agent transcript, rotate it. Better
patterns: SSH keys with `BatchMode`, scoped sudoers instead of passwords, and
on-box secret generation moved by scp. Password rotation flow that never
echoes the secret is documented in issue #15.

**20. Close issues with outcomes, not activity.**
The workflow that made this repo accumulative: file the issue with
acceptance criteria → measure before/after → commit with the evidence in the
body → close with an outcome comment (including honest "premise disproven"
closes — #8's NVDEC and the TRT INT8 gap both saved future sessions by being
closed *with data* rather than left open in hope).
