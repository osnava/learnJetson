# Troubleshooting Runbooks

Field-tested recovery guides for the Jetson Orin Nano Developer Kit (JetPack 6.x / L4T R36.x).
Each runbook follows the same anatomy: **Symptom → Fast diagnosis → Fix ladder → Root cause →
Prevention → Sources** — so both humans and AI agents can execute them step by step.

## Index

| # | Symptom / Topic | Runbook | Status |
|---|---|---|---|
| 1 | Black screen with blinking cursor, no network, after a storage change or failed boots | [boot-black-screen-blinking-cursor.md](boot-black-screen-blinking-cursor.md) | ✅ Verified in the field |
| 2 | Migrating the root filesystem from SD card to NVMe SSD without reflashing | [sd-to-nvme-rootfs-migration.md](sd-to-nvme-rootfs-migration.md) | ✅ Verified in the field |
| 3 | Locking clocks + fan for sustained MAXN_SUPER inference (and the three services that steal the fan back) | [lock-clocks-fan-maxn-super.md](lock-clocks-fan-maxn-super.md) | ✅ Verified in the field |

## The three boot-failure tripwires (know these first)

If your Jetson shows a black screen after **any** storage change or crash loop, one of these is
almost always the cause. Details and fixes in runbook #1:

1. **"Add new devices to top or bottom of boot order" is factory-set to `Top`** — inserting any
   new disk (NVMe, USB) auto-creates a boot entry *above* your working boot disk.
2. **`BOOTAA64.efi` on an ext4 partition is unloadable** — UEFI/L4T expects a FAT ESP; a stray
   copy on ext4 hangs the boot *without falling through* to the next entry.
3. **3 consecutive failed boots flip "OS chain A status" to `Unbootable`** — the board then
   boots a recovery path that looks like a dead machine (black screen, no network).

**Golden rule:** when a boot hangs, do **not** power-cycle it repeatedly — every failed cycle
feeds tripwire #3. Fix it from the UEFI menu instead.

## Adding a new entry

Copy [_template.md](_template.md), fill every section, add a row to the index above.
