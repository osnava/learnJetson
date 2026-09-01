# Runbook: Lock clocks and fan for sustained MAXN_SUPER performance

> **Verified in the field:** 2026-09-01, Jetson Orin Nano Engineering Reference Developer Kit Super
> (P3767-0005), JetPack 6.2.2 / L4T R36.5.2, kernel 5.15.199-tegra. Validated with a 10-minute
> continuous TensorRT inference run (issue #6).

## Goal

Under sustained inference load, stock DVFS and fan heuristics can cost performance. This runbook
pins the machine at its MAXN_SUPER envelope (25 W) with:

- CPU locked at 1.728 GHz on all 6 cores, idle states disabled
- GPU locked at 1.02 GHz (the Super unlock), EMC at max with frequency override
- Fan pinned at PWM 255 (100%), all dynamic fan control disabled
- All of it **persistent across reboots**

## The end state (what "locked" looks like)

```
$ sudo jetson_clocks --show
cpu0:  Online=1 Governor=schedutil MinFreq=1728000 MaxFreq=1728000 CurrentFreq=1728000 IdleStates: WFI=0 c7=0
...   (all 6 cores identical)
GPU MinFreq=1020000000 MaxFreq=1020000000 CurrentFreq=1020000000
EMC MinFreq=204000000 MaxFreq=3199000000 CurrentFreq=3199000000 FreqOverride=1
FAN Dynamic Speed Control=disabled hwmon0_pwm1=255
NV Power Mode: MAXN_SUPER
```

## Why naive persistence breaks: three fan claimants

`jetson_clocks --fan` does the right thing for the current session: it stops `nvfancontrol`,
switches the Tj thermal zone to `user_space` policy (disarming the **kernel thermal governor**),
and writes 255 to every fan PWM node. But at the next boot, up to three things fight you back:

1. **Kernel thermal governor** — if nothing sets the Tj zone to `user_space`, the kernel re-manages
   the fan (observed: PWM dropping 255 → 88 at ~50 °C).
2. **`nvfancontrol.service`** — enabled by default, restarts on its own schedule
   (`Restart=on-failure`, plus journal evidence of re-activation seconds after a manual stop).
3. **jtop (`jetson-stats` 4.3.2)** — the killer nobody suspects. Its `FanService.initialization()`
   **re-applies the fan profile saved in `/usr/local/jtop/config.json` at every service start**
   (a saved `cool` profile literally runs `systemctl start nvfancontrol`), and the daemon
   **rewrites that config from observed state** — a manually-edited `speed: 100` came back as
   `34.509…` (= PWM 88) after one boot.

The fix below disarms all three, then owns the fan from a single systemd unit.

## Step by step

### 1. Apply now

```bash
sudo jetson_clocks --fan
```

### 2. Create the boot service

No `jetson_clocks.service` ships with L4T R36.5, and the pre-created `/etc/systemd/system/jetson_clocks.service`
may be a mask (symlink to `/dev/null`) — remove it first if so.

```bash
sudo tee /etc/systemd/system/jetson_clocks.service > /dev/null <<'EOF'
[Unit]
Description=NVIDIA jetson_clocks --fan (lock MAXN_SUPER clocks, pin fan 100%)
After=nvfancontrol.service nvpmodel.service

[Service]
Type=oneshot
Environment=HOME=/root
ExecStart=/usr/bin/jetson_clocks --fan
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable jetson_clocks.service
```

Notes:
- `Environment=HOME=/root` — the script writes `~/.jetsonclocks_fan.txt`; systemd units have no `HOME`.
- **Do NOT add `jtop.service` to `After=`** — `jtop.service` itself has `After=multi-user.target`,
  which forms an ordering cycle with a unit that is `WantedBy=multi-user.target`. systemd logs
  `Found ordering cycle … Job jetson_clocks.service/start deleted` and silently never runs your unit.

### 3. Disarm nvfancontrol

```bash
sudo systemctl disable --now nvfancontrol
```

`disable` (removing the only activation path, `multi-user.target.wants`) is required — merely
stopping it loses to its restart behavior at next boot.

### 4. Disarm jtop's fan management

Stop the service first, **then** edit the config (the running daemon rewrites it from observed
state):

```bash
sudo systemctl stop jtop
sudo python3 - <<'EOF'
import json
p = "/usr/local/jtop/config.json"
c = json.load(open(p))
c.pop("fan", None)          # remove the whole fan section — jtop then never touches it
json.dump(c, open(p, "w"), indent=4)
EOF
sudo systemctl start jtop
```

Keep `"jetson_clocks": {"boot": true}` if present — jtop's own boot-time run of plain
`jetson_clocks` (without `--fan`) never writes the fan and is harmless redundancy.

### 5. Verify

```bash
systemctl is-active jetson_clocks.service   # active
systemctl is-active nvfancontrol            # inactive (and disabled)
sudo jetson_clocks --show | grep -E 'FAN|GPU Min|Power'
# → FAN Dynamic Speed Control=disabled hwmon0_pwm1=255
```

Reboot and re-check several minutes after boot (jtop runs its delayed clock job ~43 s after its
start; the fan must still read 255 then).

## Validation record (2026-09-01)

Workload: `trtexec --loadEngine=/ssd/racetrack_model.engine --duration=600` (TensorRT 10.3,
YOLO-style engine, input 1x3x416x416) — **PASSED**, 318.2 qps, mean GPU compute 3.14 ms,
599.4 s of continuous inference. Telemetry: 130 samples at 5 s intervals (CPU/GPU/SOC/Tj temps,
per-core CPU freq, GPU/EMC freq via devfreq + bpmp debugfs, fan PWM). The first ~2 minutes ran
two engines concurrently (worst case); the rest ran solo.

| Metric | Result |
|---|---|
| GPU clock | 1.02 GHz in **130/130** samples — zero dips |
| EMC clock | 3.199 GHz in 130/130 samples |
| CPU clock | 1.728 GHz, all 6 cores, 130/130 samples |
| Fan PWM | 255 constant |
| GPU temp | max **54.0 °C** (mean 52.9) |
| CPU temp | max 52.7 °C (mean 51.5) |
| Tj | max 54.0 °C — far below the ~97 °C throttle region |
| Module power | VDD_IN ≈ 15.5 W (within the 25 W MAXN_SUPER envelope) |

Safe operating temps: with the fan pinned at 100 %, the board sits ~43 °C below thermal throttle
under full sustained inference — enormous margin even in a warm enclosure.

## Revert

```bash
sudo systemctl disable --now jetson_clocks.service   # unlock clocks (back to DVFS)
sudo rm /etc/systemd/system/jetson_clocks.service
sudo systemctl enable --now nvfancontrol             # dynamic fan control back
# jtop: re-add a fan section via the jtop UI (it rewrites its own config)
```

## Sources

- NVIDIA Jetson docs — clock frequency control and `jetson_clocks` (`SD/Power and Performance`)
- Field evidence on this board: `journalctl -b -u nvfancontrol`, `journalctl -b -u jtop`
  (jtop `FanService` log lines: "Found nvfancontrol.service", "Restart nvfancontrol With profile …"),
  jtop 4.3.2 source `/usr/local/lib/python3.10/dist-packages/jtop/core/fan.py`
