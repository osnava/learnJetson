# agent/inventory.md — template

> Copy to `agent/inventory.md` (gitignored) and fill with **your** values.
> For `jetson.sh`, the same values can live in `agent/inventory.sh` as exports:

```bash
# agent/inventory.sh  (gitignored)
export JETSON_USER="youruser"
export JETSON_SUBNET="192.168.1"        # your LAN prefix (no trailing dot)
export JETSON_ROOTFS="/dev/nvme0n1p1"   # expected root device for health checks
```

## Machine facts

| Fact | Value |
|---|---|
| Model / module / carrier | Jetson Orin Nano Dev Kit (P3767-0005 / P3768) |
| JetPack / L4T / kernel | 6.2.2 / R36.5.2 / 5.15.199-tegra |
| Boot design | SD = boot disk (extlinux), rootfs on NVMe `root=PARTUUID=<nvme-p1-partuuid>` |
| Data disk | NVMe p2 → `/ssd` (UUID `<p2-uuid>`), 16G swapfile `/ssd/16GB.swap` |
| SD ESP (boot) | `<sd-esp-uuid>` (vfat) |
| Wi-Fi MAC (for DHCP reservation) | `<mac>` |
| USB-C fallback (always) | `192.168.55.1` (RNDIS) + COM serial device |
| Serial console (if you buy the adapter) | Button Header: TX→pin 3, RX→pin 4, GND→pin 7, 115200 8N1 |

## UEFI state (should always be)

- `Add new devices to top or bottom of boot order` = **Bottom**
- `OS chain A status` = **Normal**
- `L4T Boot Mode` = **ExtLinux**
- BootOrder starts with the SD entry; the NVMe entry sits last

## Maintenance reminders

- After any kernel-package `apt upgrade`, sync `/boot/Image` + `/boot/initrd` to the SD
  (see the migration runbook §maintenance).
- Sudo for agents: prefer a scoped sudoers drop-in over sharing the main password.
