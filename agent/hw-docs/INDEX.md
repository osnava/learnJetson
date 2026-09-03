# INDEX.md — hardware questions: route, then answer with a citation

> The agent's map of the hardware-document corpus for this machine
> (P3766 dev kit = P3767 module + P3768 carrier, JetPack 6.2.2 / L4T R36.5.2).
> `./fetch.sh` materializes the documents as markdown in `md/` (gitignored).
> Downloads are ~9 MB core, ~107 MB `--full` (SoC TRM + carrier schematics);
> budget ~160 MB on disk after `--full` — PDFs, markdown and figures together.
> The corpus lives **on the PC the agent operates from** — nothing is
> fetched to or stored on the Jetson itself. Originals cache in `pdf/`.
> Figures are extracted to `md/images/<doc>/` and linked inline where they
> appear (a cull pass drops repeated decorations and sub-120 px fragments,
> and collapses duplicate copies to one — that was 97% of the TRM's raw
> haul; repeat *links* survive, repointed). Identify a figure by
> the extracted picture-text next to its link; **never assert what a
> figure shows unless your harness truly renders images** — for humans,
> the cached PDF at the cited page is authoritative. Every converted page
> starts with a `<!-- p.N -->` anchor — cite answers as `doc §section
> (p. N)`.

## Protocol (mirrored in AGENTS.md)

1. Route the question through the table below; read/grep that section.
2. Answer with `doc §section (p. N)` and quote the load-bearing line.
3. Document missing? Run `./fetch.sh` before guessing. Datasheet absent?
   It is NVIDIA-login-gated — one-time manual download (fetch.sh prints how).
4. Authority order: **Data Sheet / Carrier Board Spec / TRM** →
   Jetson Linux Developer Guide (online, r36.x) → NVIDIA forums/blogs —
   leads only, never the sole source.
5. Pin names differ between Data Sheet, TRM, and design files. Reconcile
   with `orin-pin-function-names.md` before trusting any pin answer.

## Routing table

| Question about | Where (in `md/`) | Section |
|---|---|---|
| GPU / CPU / memory subsystem specs | `datasheet.md` | §2.1–2.4 |
| Video decode & encode in silicon | `datasheet.md` | §2.9 + Table 2-5 |
| Module interfaces (PCIe, USB, UART, I2C, GPIO counts) | `datasheet.md` | §2.10 |
| Power rails, domains, sequencing, power management | `datasheet.md` | Ch. 3 (§3.1–3.8) |
| Module pinout / GPIO ball map | `datasheet.md` | §4.3–4.4 |
| Electrical limits (abs-max, recommended, logic levels) | `datasheet.md` | §5.1–5.2 |
| USB / Ethernet / DisplayPort / M.2 Key E & M slots | `devkit-carrier-spec.md` | §2.1–2.5 |
| Camera connectors (MIPI CSI-2) | `devkit-carrier-spec.md` | §3.2 |
| 40-pin expansion header | `devkit-carrier-spec.md` | §3.3 |
| Button header (debug UART, reset, recovery, sleep/wake, LED) | `devkit-carrier-spec.md` | §3.4, Table 3-4 |
| Optional CAN header | `devkit-carrier-spec.md` | §3.5 |
| Optional PoE / backpower headers | `devkit-carrier-spec.md` | §3.9 |
| Fan connector pinout | `devkit-carrier-spec.md` | §3.6 |
| Fan drive type (PWM + tach), fan curves, quiet/cool modes | `orin-nx-nano-design-guide.md` · `orin-thermal-design-guide.md` | §12.5 · §5.2 |
| DC power jack / RTC coin cell | `devkit-carrier-spec.md` | §3.8, §3.7 |
| Pinmux defaults, per-ball alternate functions | `orin-pinmux.*.csv` | sheets `…Pinmux_DP` / `…Pinmux_HDMI` |
| Pin-name differences across the docs | `orin-pin-function-names.md` | whole doc |
| Heatsink, airflow, thermal specs | `orin-thermal-design-guide.md` | §2.1, §4.2–4.4 |
| Max operating temp, HW throttling, shutdown temp | `orin-thermal-design-guide.md` | §5.3–5.5 |
| UART / I2C / SPI / CAN electrical + routing rules | `orin-nx-nano-design-guide.md` | §12.1–12.4 |
| DP / eDP / HDMI design + routing | `orin-nx-nano-design-guide.md` | §9.1–9.2 |
| Boot device selection / USB force recovery (hardware) | `orin-nx-nano-design-guide.md` | §3.1–3.2 |
| Boot straps — pins pulled/driven high at power-on | `orin-nx-nano-design-guide.md` | §13.3 (Table 13-1); pull-ups §13.1–13.2 |
| Dev-kit specifics (button MCU, USB hub, level shifters, PoE) | `orin-nx-nano-design-guide.md` | §4.1–4.4 |
| SoC registers, pad control, controller internals | `orin-trm.md` (`--full`) | grep — 7,000+ pages |
| Carrier schematics — net names, refdes, per-sheet blocks | `devkit-carrier-schematics.md` (`--full`) | grep by net or refdes |
| BOM / gerbers / layout of *this* carrier | `pdf/devkit-carrier-reference-design/` (`--full`) | — |

Two answers worth memorizing (both verified against the corpus):

- **No hardware video encoder on Orin Nano** — `datasheet.md` Ch. 1
  Overview, "HD Video → Encode" (p. 7), states it affirmatively:
  *"1080p30 Supported via CPU Cores with Software."* Cite that line, not
  the absence of an encoder block in §2.9 — §2.9.1 and Table 2-5 describe
  the NVDEC **decoder** (H.265, H.264, VP9, VP8, AV1, MPEG-4, MPEG-2,
  VC-1) and are the wrong place to argue from silence. Primary source
  behind FIELD_NOTES #9.
- **Button-header voltage domains are mixed** — §3.4 Table 3-4 (p. 28):
  debug UART pins 3/4 are 3.3 V, `SYS_RESET*` (pin 8) and
  `FORCE_RECOVERY*` (pin 10) are 1.8 V, and the 5 V domain covers the
  sleep/wake LED on pins 1/2 (`PC_LED-`/`PC_LED+`) plus `SLEEP/WAKE*` on
  pin 12 — three different functions, not one "LED" pair. Pin 3 is the
  board's **UART2_RXD** — mind adapter-vs-board TX/RX naming (see the
  serial-console row in `../inventory.md`).

## Online-only supplements (no local copy)

- [Jetson Linux Developer Guide r36.x](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/) — boot chain, `extlinux.conf`, platform config.
- Jetson Platform Power & Performance (power-mode table, MAXN/MAXN_SUPER) — linked from [SETUP.md §6](../SETUP.md).
- Jetson AI Lab tutorials; JetPack 6.2 Super-mode announcement — linked from SETUP.md §1–2.

## Pinned versions (URLs verified 2026-09-02)

| Document | Version | Source |
|---|---|---|
| Jetson Orin Nano Series Modules Data Sheet | 1.7 (DS-11105-001) | [Download Center](https://developer.nvidia.com/embedded/downloads) — login-gated |
| Orin NX + Nano Series Design Guide | 1.5 | [direct PDF](https://developer.nvidia.com/downloads/jetson-orin-nx-series-nano-series-design-guide) |
| Orin Nano DevKit Carrier Board Spec | 1.3 (SP-11324-001) | [direct PDF](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf) |
| Orin NX + Nano Pinmux template | 1.2 | [direct xlsx](https://developer.nvidia.com/downloads/jetson-orin-nx-and-orin-nano-series-pinmux-config-template) |
| Orin NX + Nano Thermal Design Guide | 1.5 | [direct PDF](https://developer.nvidia.com/downloads/jetson-orin-nx-orin-nano-series-thermal-design-guide) |
| Pin & Function Names Guide | 1.0 (DA-11434-001) | [direct PDF](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nx/docs/jetson_orin_nx_orin_nano_pin_function_names_guide_da-11434-001_v1.0.pdf) |
| Jetson Orin Series SoC TRM | 1.2p | [direct PDF](https://developer.nvidia.com/downloads/orin-series-soc-technical-reference-manual/) |
| DevKit Carrier reference design files | A04, 2023-03-20 | [direct zip](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_reference_design_files_a04_20230320.zip) |

Refresh check: at every JetPack bump, compare versions against the
[Jetson Download Center](https://developer.nvidia.com/embedded/downloads)
(search the document title) and update this table + `fetch.sh` together.

## Converter

`convert.py` uses **pymupdf4llm** (`pip install pymupdf4llm openpyxl`):
real heading/table structure on born-digital PDFs, no ML models, no GPU.
Marker/Docling/MinerU beat it only on scanned or complex layouts, at the
cost of multi-GB model downloads; the NVIDIA corpus is born-digital.
Figures are extracted with `write_images=True` into `md/images/<doc>/`
with inline links; `_cull_figures` then removes sub-120 px fragments,
near-blanks and exact duplicates (a 36 px decoration repeated across the
TRM was 8,771 of its 8,980 raw images — without the cull, "extracted
figures" is mostly noise).
Duplicate figures are deleted from disk but keep their links, repointed
at the one surviving copy — a diagram that legitimately recurs stays
referenced in both sections. Under `--full`, `fetch.sh` also converts the
carrier schematics out of the reference-design zip into
`devkit-carrier-schematics.md`; everything in `md/` comes from the script.
Fallback if pymupdf4llm is missing: poppler `pdftotext -layout` (text
only, headings lost — the output says so). Very large PDFs (the TRM)
convert in 200-page batches so memory stays bounded and progress is visible.

**Reading converted tables:** pymupdf4llm splits multi-row table headers,
so the first `|…|` row is often partial and the *real* column names sit in
the row below it — carrier Table 3-4 renders as
`|**Pin**||**Module**||**Type/Dir**|` before the row naming all five
columns. Map columns off the second row, not the first.
