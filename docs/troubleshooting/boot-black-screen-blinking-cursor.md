# Runbook: Black screen with blinking cursor, no network

> **Verified in the field:** 2026-08-31, Jetson Orin Nano Developer Kit (P3767-0005 on P3768),
> JetPack 6.2.2 / L4T R36.5.2. Symptom appeared after an SD→NVMe migration attempt; full recovery
> achieved keyboard-only in ~20 minutes once the root cause was known.

## Symptom

- NVIDIA logo shows, then the screen goes black with only a **blinking cursor**.
- **No network** (the device does not appear on the LAN, no SSH).
- Monitor + USB keyboard attached; no serial console available.
- Typically follows a storage change (disk inserted/removed) or a series of failed boots.

## Fast diagnosis

| Evidence | Likely cause | Confidence |
|---|---|---|
| Symptom started right after inserting a disk | Boot-order hijack (tripwire 1) + possibly stray EFI file (tripwire 2) | High |
| Symptom survived disk removal and filesystem repairs | "OS chain A status: Unbootable" (tripwire 3) | High |
| A "L4T boot options … Press 0-1 within 3.0 seconds" menu flashes, then black | Launcher on a failing/recovery path | High |
| Kernel/boot messages never appear on screen | Failure is pre-kernel (UEFI/bootloader stage) | Medium |

## Fix ladder (cheapest first — stop when the system boots)

### Step 1 — The UEFI ritual (keyboard-only, fixes all three tripwires)

1. Power off; **unplug for 30 s** (cold drain also clears display warm-boot wedges).
2. Power on while **repeatedly tapping `ESC`** at the NVIDIA splash → UEFI menu.
3. `Device Manager → NVIDIA Configuration → L4T Configuration`:
   - **`OS chain A status`**: if `Unbootable` → set **`Normal`**.
   - **`L4T Boot Mode`**: set **`ExtLinux`** (not Recovery/Kernel Partition).
4. `Device Manager → NVIDIA Configuration → Boot Configuration`:
   - **`Add new devices to top or bottom of boot order`** → set **`Bottom`** (factory default is
     `Top` — this is the root cause of storage-change hijacks).
5. `F10` to save; `ESC` back; **`Save Changes and Exit`**.
6. On reboot, if the **3-second "L4T boot options"** menu appears, press **`0`** ("primary kernel").
7. The system should boot normally. If a newly inserted disk is present, also delete any stray
   boot entry for it (`Boot Maintenance Manager → Boot Options → Delete Boot Option`) — note the
   entry may regenerate on every boot while the disk is present; the `Bottom` policy is what
   makes it harmless.

### Step 2 — USB-C device-mode probe (zero cost, tells you if the OS is alive)

The Orin Nano devkit's USB-C port (data-only; power comes from the barrel jack) exposes, when
the Linux kernel finishes booting:

- a **USB serial console** (COM port on the host PC), and
- an **RNDIS USB-Ethernet adapter at the fixed IP `192.168.55.1`** — SSH-able, no DHCP needed.

Plug Jetson → PC during a boot attempt. A new COM/RNDIS device appearing means the OS boots and
the problem is display-side (try `sudo systemctl restart gdm` over SSH). Nothing appearing means
a boot-chain problem → back to Step 1 or on to Step 3.

### Step 3 — Serial console (full visibility; needs a USB-TTL adapter)

No micro-USB debug port exists on this kit. Serial is a USB-TTL cable on the **Button Header**
(the same 12-pin header as the recovery jumper):

| Adapter wire | Header pin |
|---|---|
| TX | 3 (RXD) |
| RX | 4 (TXD) |
| GND | 7 |

115200 8N1. `ESC` works over serial for UEFI, and the recovery-kernel shell it exposes can reset
the UEFI variables directly (see "Set the UEFI Variable in the Recovery Kernel Shell" in the
NVIDIA UEFI docs, below).

### Step 4 — Last resort: reflash the SD card from any PC

- No Ubuntu host or recovery mode needed: write the official **JetPack SD card image** with
  balenaEtcher (for JetPack 6.2.2, NVIDIA says to use the 6.2.1/L4T 36.4.4 image and apt-upgrade).
  ⚠️ This erases the SD — back it up first.
- Recovery-mode SDK Manager reflash exists too, but is rarely necessary for SD-boot kits.

## Root cause (the full chain, as observed)

1. Factory UEFI policy **"new devices → Top"**: inserting the NVMe auto-created a boot entry
   *above* the SD entry, so UEFI tried the new disk first.
2. That entry pointed at `/EFI/BOOT/BOOTAA64.efi` **on an ext4 partition** — unloadable (L4T
   expects a FAT ESP) → UEFI **hung without falling through** to the next boot entry.
3. Repeated hung boots + hard power-offs (≥3 consecutive failures) flipped
   **`OS chain A status` → `Unbootable`** (`RootfsStatusSlotA = 0xFF`), sending every later boot
   down a recovery path = black screen + blinking cursor, no network — even with a perfectly
   healthy SD card. This is why filesystem repairs (fsck, fstab, ESP dirty-bit) changed nothing:
   the broken state lives in UEFI NVRAM, not on disk.

## Prevention

- Keep **"Add new devices…" = `Bottom`** forever.
- Never place `BOOTAA64.efi` on an ext4 partition; the launcher belongs on the FAT ESP of the
  boot disk. Disable stray `extlinux.conf` files on non-boot disks (rename to `.disabled`).
- When a boot hangs: **fix via the UEFI menu; do not power-cycle repeatedly** (each failed cycle
  feeds the Unbootable tripwire).
- A single successful full boot clears the retry counters (marked by NVIDIA userspace services).

## Verification after recovery

From the host, sweep the subnet for SSH (e.g. `agent/jetson.sh find`), then check:
`findmnt -n -o SOURCE /` (expected rootfs), `swapon --show`, `systemctl --failed`,
`efibootmgr` (BootOrder starts with your boot disk).

## Sources

- NVIDIA Jetson Linux R36.5 UEFI docs — OS chain status & boot mode fix, recovery shell variables:
  https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/Bootloader/UEFI.html
- Exact-symptom thread (black screen + blinking cursor after update; the `0` keypress fix):
  https://forums.developer.nvidia.com/t/boots-into-black-screen-with-a-blinking-cursor-after-update-reboot/284063
- BOOTAA64.efi cannot load from ext4: https://forums.developer.nvidia.com/t/load-bootaa64-efi-from-a-different-fat-partition/229379
- 3-failed-boots → slot marked Unbootable (NVIDIA staff explanation):
  https://forums.developer.nvidia.com/t/l4tlauncher-would-like-to-avoid-being-bricked-when-both-a-b-slots-are-marked-as-unbootable/288078
- "Add new devices → Bottom" recommended by NVIDIA staff (same platform):
  https://forums.developer.nvidia.com/t/boot-order-issue-on-orin-nano/305599
- Confirmation that Top is default and Bottom works:
  https://forums.developer.nvidia.com/t/uefi-boot-order-setting-append-storage-devices-to-the-top-bottom-boot-order/315888
- L4TLauncher scans all storage for extlinux.conf ("first found wins"):
  https://forums.developer.nvidia.com/t/how-do-i-resolve-error-processextlinuxconfig-unable-to-find-partition-info/325908
- Serial console via Button Header + USB-C device mode (RNDIS 192.168.55.1):
  https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html
  https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html
- JetPack 6.2.2 SD image guidance: https://developer.nvidia.com/embedded/jetpack-sdk-622
