# AGENTS.md — Operating this Jetson with an AI agent

This file is the **rulebook for any AI agent** (Claude Code, ZCode, Cursor, etc.) that operates
the Jetson Orin Nano described by this repository. It exists so "use an AI agent on my Jetson"
is a safe, repeatable statement — for the owner and for anyone adopting this repo.

**The kit, in reading order for a fresh session:** this rulebook →
[`inventory.md`](inventory.md) (machine facts, gitignored) →
[SETUP.md](SETUP.md) if you're bringing up a **new** Jetson with this kit →
[FIELD_NOTES.md](FIELD_NOTES.md) for the *why* behind these rules, with
sources → [`hw-docs/INDEX.md`](hw-docs/INDEX.md) the moment a question
touches hardware → the [runbooks](../docs/troubleshooting/README.md) when
something breaks.

## The machine (see `inventory.md` — gitignored — for real values)

- Jetson Orin Nano Developer Kit, JetPack 6.2.2 (L4T R36.5.2), kernel `5.15.199-tegra`.
- **Boot design:** SD card = boot disk (L4TLauncher → `extlinux.conf`), **rootfs on NVMe**
  (`root=PARTUUID=…`), data disk `/ssd` on NVMe p2. See
  [the migration runbook](../docs/troubleshooting/sd-to-nvme-rootfs-migration.md) before
  touching anything storage-related.
- Access: SSH with a key (BatchMode-friendly). DHCP address varies per boot.

## Hardware questions — answer from primary docs

Anything about pins, voltage domains, connectors, power rails, video
encode/decode engines, thermal limits, registers: route it through
[`hw-docs/INDEX.md`](hw-docs/INDEX.md) and answer from the fetched
markdown in `hw-docs/md/`, citing `doc §section (p. N)`.

- Corpus not fetched yet? Run `hw-docs/fetch.sh` (~9 MB core; `--full`
  adds the SoC TRM + carrier schematics). The data sheet itself is
  NVIDIA-login-gated — one-time manual download, the script prints how.
  The corpus lives on **this PC** — operator-side knowledge; nothing is
  fetched to or stored on the Jetson.
- Authority order: **Data Sheet / Carrier Board Spec / TRM** → Jetson
  Linux Developer Guide (online) → NVIDIA forums/blogs (leads only,
  never the sole source).
- Cross-check pin answers against the Pin & Function Names guide — pin
  naming differs between Data Sheet, TRM, and design files.

## Discovery — always in this order

1. `./jetson.sh find` — sweeps the LAN subnet for SSH and probes the USB-C fallback.
2. **USB-C fallback:** if the OS is up but Wi-Fi isn't cooperating, the device-mode link is
   always at **`192.168.55.1`** (RNDIS, works with a plain USB-C data cable, no network infra).
3. Nothing found → the machine is off or pre-boot-dead → **escalate to the human**
   (see [the black-screen runbook](../docs/troubleshooting/boot-black-screen-blinking-cursor.md)).

## Allowed without asking

- Read-only diagnostics: `findmnt`, `lsblk`, `df`, `swapon`, `systemctl status/--failed`,
  `journalctl`, `docker ps/images/info`, `efibootmgr` (read), `nvtop/jetson_stats`.
- Service control via the guarded surface: `systemctl restart|start|stop docker gdm`,
  starting/stopping the project's own containers. These work non-interactively as
  `sudo -n <cmd>` via the scoped sudoers drop-in (`/etc/sudoers.d/agent-toolkit`,
  issue #15 — also covers `journalctl` and `docker`); no shared passwords.
  Whitelisted verbs only: any other sudo still needs the owner.
- Deploying/running code under `$HOME` and the project's own directories.

## Ask the human first

- Any `apt upgrade` **that includes kernel packages** (linux-image*) — the kernel-sync caveat
  of the boot design applies (see migration runbook §maintenance).
- Anything touching `/boot/extlinux/extlinux.conf`, `/etc/fstab`, partition tables, UEFI
  variables (`efibootmgr` writes), or `/ssd/docker`.
- Installing packages, opening network ports, anything that changes the machine's identity.

## Forbidden — ever

- **Never power-cycle or reboot a hung boot repeatedly.** Three consecutive failed boots flip
  UEFI's "OS chain A status" to `Unbootable` and the machine looks dead (black screen). Fix
  boot problems from the UEFI menu, not the power button.
- Never place `BOOTAA64.efi` on an ext4 partition or leave a live `extlinux.conf` on a
  non-boot disk (launcher scans all storage — first found wins).
- Never edit `fstab`/`extlinux.conf` without first copying it to a `.bak` sibling.
- Never touch QSPI/bootloader partitions, and never run a full reflash without explicit
  instruction — data loss is the default outcome of flashing tools.

## Escalate to a human at the keyboard (agents have no hands)

- Anything physical: UEFI menus (`ESC` at splash), jumpers/recovery mode, disk insertion,
  display cables. Provide exact keystrokes from the runbooks and wait.

## Protocol after any change

1. Verify with `./jetson.sh status <ip>` (rootfs source, mounts, swap, docker, failed units).
2. For boot-path changes: one warm reboot **and** one cold power cycle, both verified over SSH,
   before declaring success.
3. Log what you did and why (one line per action, in the session notes or PR description).

## Tooling in this folder

| File | Purpose |
|---|---|
| `jetson.sh` | find / ssh / status / health / logs / dropcache — the agent's hands |
| `inventory.md` | real IPs/MACs/UUIDs (**gitignored — never commit**) |
| `inventory.example.md` | template for the above |
| `hw-docs/INDEX.md` | hardware-question routing table: question → doc §section (p. N) |
| `hw-docs/fetch.sh` | materialize the hardware corpus as markdown in gitignored `hw-docs/md/` |
| `hw-docs/grade.py` | citation grader — verify an answer's `doc §section (p. N)` + quote really resolves in the corpus (exit 1 = a citation fails verification; 2 = corpus or document not fetched — not the agent's fault) |
| `hw-docs/check.sh` | corpus linter, entrypoint to `check.py` (CI runs it on every push) — INDEX routing rows, version pins, memorized answers, conversion smoke, URL HEAD checks; absent docs report SKIP, never PASS |
| `hw-docs/eval/questions.yaml` | golden question set (issue #24): 26 fixed questions with ground truth, ~40% unanswerable-with-redirect; every answerable item's citation re-verified by `hw-docs/test_questions.py` (CI runs it) — feeds the cold-session runner |
| `launch_vllm.sh` | Cosmos-Reason2 vLLM launcher — stream to the Jetson, run by path ([runbook](../docs/cosmos-reason2-vllm.md)) |
| `cosmos-env.example` | template for the Jetson's `~/.cosmos-env` (real file **gitignored**) |

## Repo hygiene on Windows (line endings + exec bits)

Line endings are governed by `.gitattributes` (`* text=auto eol=lf`): every
clone keeps LF working trees, so Windows/WSL/Linux copies never produce
phantom whole-file diffs. Two Windows-specific hazards remain:

- **The executable bit doesn't exist on NTFS.** A file rewritten by a Windows
  tool loses `mode 100755` (this bit `agent/jetson.sh` once). Check with
  `git ls-files -s agent/jetson.sh`, restore with
  `git update-index --chmod=+x agent/jetson.sh` — git stores the bit even
  though the Windows filesystem can't show it.
- **Don't edit the same files from WSL and Windows simultaneously** — the
  WSL clone is retired anyway (Windows clone is primary).
