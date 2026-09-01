# Runbook: Move the root filesystem from SD card to NVMe SSD (no reflash)

> **Verified in the field:** 2026-08-31, Jetson Orin Nano Developer Kit, JetPack 6.2.2.
> Design goal (GitHub issue "Move root filesystem from SD card to NVMe SSD"): keep all data and
> packages, avoid SDK Manager/reflash entirely.

## Choose your approach first

| Approach | Preserves data | Needs host PC / recovery mode | Wipes disk | Notes |
|---|---|---|---|---|
| **A. SD-boot-disk + NVMe rootfs** (this runbook) | ✅ | ❌ | ❌ | Boot files stay on the SD; kernel must be synced after upgrades (below) |
| B. Official reflash to NVMe (`l4t_initrd_flash.sh --external-device nvme0n1p1` / SDK Manager) | ❌ | ✅ Ubuntu host + recovery jumper | ✅ | Fresh install; proper FAT ESP on the NVMe. [NVIDIA docs](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/FlashingSupport.html), [RidgeRun guide](https://developer.ridgerun.com/wiki/index.php/How_to_flash_and_boot_a_Jetson_from_NVMe_SSD) |
| C. Write the SD image directly to the NVMe (PC + enclosure) | ❌ | ✅ enclosure | ✅ | [Community guide](https://www.reddit.com/r/JetsonNano/comments/1hth1vo/booting_jetson_orin_nano_super_from_ssd/) |
| D. Native NVMe boot with its own FAT ESP (shrink p1, add ESP, NVMe first) | ✅ | ❌ | ❌ | Most steps; only worth it if you must remove the SD entirely |

Approach A is the fastest data-preserving path and is what this repo runs.
**Read [boot-black-screen-blinking-cursor.md](boot-black-screen-blinking-cursor.md) before
touching storage** — approach A is exactly the scenario that triggers the boot-order hijack if
the UEFI policy is still factory-default.

## Approach A, step by step

### 0. Preconditions

- System healthy, booted from SD, SSH working.
- UEFI → `Boot Configuration → "Add new devices to top or bottom of boot order"` = **`Bottom`**
  (do this **before** the NVMe goes in; see the black-screen runbook).
- Backup anything irreplaceable (`/root/migration-backup/` style tar).

### 1. Prepare the NVMe (one data disk, one rootfs)

```
GPT:
  p1  ext4  ~170G   future rootfs (clone of the SD rootfs)
  p2  ext4  rest    data disk, mounted at /ssd (docker, models, datasets, swapfile)
```

Clone the rootfs while running from SD (rsync with `-aHAXx` from a live/secondary environment,
or `dd`/clone the SD from a PC). Copy `/ssd` data onto p2 and create a swapfile (e.g. 16G) there.

### 2. Fix `/etc/fstab` **on the clone** (NVMe p1)

```
/dev/root            /          ext4  defaults                       0 1
UUID=<p2-uuid>       /ssd/      ext4  defaults,nofail                0 2
/ssd/16GB.swap       none       swap  sw,nofail                       0 0
UUID=<SD-ESP-uuid>   /boot/efi  vfat  defaults                        0 1
```

`nofail` on `/ssd` keeps a missing data disk from dropping you into emergency mode.

### 3. Two hard rules (violating either bricks boot — see the black-screen runbook)

- **Never** copy `BOOTAA64.efi` onto the ext4 NVMe partition. The launcher lives only on the FAT
  ESP of the boot disk (the SD, partition 10 on the devkit layout).
- The NVMe must **not** carry its own live `/boot/extlinux/extlinux.conf` — otherwise the
  launcher (which scans all storage, first-found-wins) may boot the wrong config. Rename it:
  `mv /boot/extlinux/extlinux.conf /boot/extlinux/extlinux.conf.disabled`.

### 4. Dual-label the SD's extlinux (fallback-safe)

On the **SD card's** `/boot/extlinux/extlinux.conf`:

```
TIMEOUT 30
DEFAULT primary

LABEL primary
      MENU LABEL primary kernel
      LINUX /boot/Image
      INITRD /boot/initrd
      APPEND ${cbootargs} root=/dev/mmcblk0p1 rw rootwait rootfstype=ext4 ... console=tty0

LABEL nvme
      MENU LABEL NVMe rootfs
      LINUX /boot/Image
      INITRD /boot/initrd
      APPEND ${cbootargs} root=PARTUUID=<NVMe-p1-PARTUUID> rw rootwait rootfstype=ext4 ... console=tty0
```

(The full `APPEND` line = the factory line with only `root=` changed. Keep a backup:
`cp extlinux.conf extlinux.conf.bak`.)

Boot once with `DEFAULT primary` (verification boot), then flip `DEFAULT nvme` and reboot.

### 5. Verify

```bash
findmnt -n -o SOURCE /            # → /dev/nvme0n1p1
cat /proc/cmdline                 # → root=PARTUUID=<...>
df -h /ssd && swapon --show
systemctl --failed                # → empty
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Also verify one full **cold power cycle** before calling it done.

## The one maintenance caveat (approach A)

The boot kernel comes from the **SD's** `/boot`, but `apt upgrade` on the NVMe rootfs installs
new kernels to the **NVMe's** `/boot`, which the boot path ignores. After any kernel-package
upgrade, sync them or the next boot pairs a new rootfs with an old kernel (module mismatch):

```bash
sudo mount /dev/mmcblk0p1 /mnt/sd && sudo cp /boot/Image /boot/initrd /mnt/sd/boot/ && sudo umount /mnt/sd
```

(If that ever feels fragile, approach D removes the SD entirely.)

## Sources

- No-host SD→NVMe clone (approach A precedent): https://forums.developer.nvidia.com/t/blog-boot-from-nvme-without-using-sdkmanager-or-external-ubuntu-pc-a-solution-that-works/252757
- SSD install/boot tutorial: https://jetsonhacks.com/2023/05/30/jetson-orin-nano-tutorial-ssd-install-boot-and-jetpack-setup/
- Official flashing support (approach B): https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/FlashingSupport.html
- initrd flash to NVMe (approach B): https://developer.ridgerun.com/wiki/index.php/How_to_flash_and_boot_a_Jetson_from_NVMe_SSD
- extlinux scanning ambiguity ("first found wins"): https://forums.developer.nvidia.com/t/how-do-i-resolve-error-processextlinuxconfig-unable-to-find-partition-info/325908
- Boot-order hijack + black screen (what goes wrong): [boot-black-screen-blinking-cursor.md](boot-black-screen-blinking-cursor.md)
