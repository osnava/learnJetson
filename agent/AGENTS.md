# AGENTS.md — Operating this Jetson with an AI agent

This file is the **rulebook for any AI agent** (Claude Code, ZCode, Cursor, etc.) that operates
the Jetson Orin Nano described by this repository. It exists so "use an AI agent on my Jetson"
is a safe, repeatable statement — for the owner and for anyone adopting this repo.

## The machine (see `inventory.md` — gitignored — for real values)

- Jetson Orin Nano Developer Kit, JetPack 6.2.2 (L4T R36.5.2), kernel `5.15.199-tegra`.
- **Boot design:** SD card = boot disk (L4TLauncher → `extlinux.conf`), **rootfs on NVMe**
  (`root=PARTUUID=…`), data disk `/ssd` on NVMe p2. See
  [the migration runbook](../docs/troubleshooting/sd-to-nvme-rootfs-migration.md) before
  touching anything storage-related.
- Access: SSH with a key (BatchMode-friendly). DHCP address varies per boot.

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
  starting/stopping the project's own containers.
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
