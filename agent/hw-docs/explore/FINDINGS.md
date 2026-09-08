# Docling spike — findings (#27)

Converted the fetched corpus with plain `docling convert` (standard
pipeline, this machine's GPU, models pre-cached), `--to md --to json`,
`--image-export-mode placeholder`. Outputs in this directory (gitignored);
this file is the deliverable. Judged natively — the pymupdf-tuned harness
is a reference point, not the target.

## Runtime

| Document | Pages | Time | Rate |
|---|---|---|---|
| devkit-carrier-spec | 38 | 69 s | 1.8 s/pp |
| orin-nx-nano-design-guide | ~108 | 148 s | 1.4 s/pp |
| orin-thermal-design-guide | ~45 | 50 s | 1.1 s/pp |
| orin-pin-function-names | 23 | 36 s | 1.6 s/pp |
| datasheet | ~36 | 75 s | 2.1 s/pp |
| orin-trm (sample pp. 120–139, dense) | 20 | 59 s | ~3 s/pp |

Full core corpus ≈ 6.5 min one-time. TRM extrapolates to **≈ 7.3 h**
(8,783 pp); `--no-tables` saves almost nothing (54 s vs 59 s on the
sample — the layout model dominates, table structure is nearly free), so
there is no cheap "text-only" TRM shortcut. TRM must be a background,
one-time grind.

## Table fidelity — the acid tests, before → after

**Carrier Table 3-4 (button header)** — pymupdf broke cells with `<br>`
and joined words:

```
|1|–|–|PC_LED-: Connects to LED Cathode to indicate System<br>Sleep/Wake (Off when system in sleepmode)|Input, 5V|
|8|SYS_RESET*|239|Temporarilyconnectpins 7 and 8 to reset system|Input, 1.8V|
```

docling, clean sentences, dehyphenated, real spaces:

```
| 1 | - | - | PC_LED-: Connects to LED Cathode to indicate System Sleep/Wake (Off when system in sleep mode) | Input, 5V |
| 8 | SYS_RESET* | 239 | Temporarily connect pins 7 and 8 to reset system | Input, 1.8V |
```

**Carrier Table 3-3 pin 8** — still wrapped (the PDF itself wraps the
cell), but as a space instead of markup:
`GP70_UART1_T<br>XD_BOOT2_STR<br>AP` → `GP70_UART1_T XD_BOOT2_STR AP`.
Token reconstruction is a source-PDF problem, not a converter problem;
either way it stays greppable after whitespace normalization.

**Design guide Table 12-7 (merged pin-type cell)** — pymupdf split
`CMOS –|…|1.8V` across rows; docling gives one complete row:

```
| 99 | UART0_TXD | GP32_UART2_TXD | UART 0 Transmit | UART general (i.e. M.2 Key E) | Output | CMOS - 1.8V |
```

**Multi-row split headers** — carrier Table 3-4's
`|**Pin**||**Module**||**Type/Dir**|` shape (which forced the "map
columns off the second row" caveat in INDEX) is gone; docling emits one
proper header row: `Pin # | Module Pin Name | Module Pin # |
Usage/Description | Type/Dir Default`.

**Pin-names giant Table 8** — 260-row table survives row-intact and
unwrapped: `| 203 | UART1_TXD | UART1_TXD | GP70_UART1_TXD_BOOT2_STRAP |`.

## Headings

Section numbers survive as real markdown headings at sane levels
(`## 3.4 Button Header`, `## 12.5 Fan`; 39 headings in the carrier spec).
Section-token citation addressing works structurally — no heading-floor
guessing needed.

## Page provenance — the substrate change

The JSON carries per-object provenance:

- table-level: `prov: [{page_no: 28, bbox: …}]` on the Table 3-4 object —
  **exactly the PDF page the golden set cites for it**
- text/caption-level: `page_no` + bbox + charspan on every text item
- picture-level: `page_no` + bbox on all 20 pictures
- `pages` dict: per-page geometry

This is the important structural difference: a citation can address an
*object with provenance* (this table, that heading, page 28) instead of a
regex over a text stream. The docling grader resolves pages structurally; the
`<!-- p.N -->` anchors and the section-span heuristics both become
unnecessary machinery rather than the contract.

## Running headers / footers

Old carrier md inlines 74 running header/footer lines; the new md has 1
(title page). Docling classifies page furniture separately from body
content. The p. 29-vs-p. 28 bug class — running headers poisoning
section spans — disappears structurally, not by tuning.

## Figures

`--image-export-mode placeholder` emits **zero** image links in the md;
the JSON still knows every picture's page and bbox. The pymupdf problem
(8,980 raw images needing a cull pass) never arises. Trade-off: no
"picture text" side-channel — consistent with INDEX's existing rule
(never assert what a figure shows; the PDF is authoritative for humans).

## Artifacts to know

- **Prose identifiers get backslash-escaped** (`M\_TTCAN`) while table
  cells stay clean (`GP32_UART2_TXD`). Any lookup layer must normalize
  `\_` → `_`.
- Paragraphs export as long single lines (fine for grep/reading).
- Cell-level provenance is empty; provenance lives on the object (table,
  text, picture) — sufficient for page-exact citations.
- Table-matching logs `MatchingPostProcessor: Orphan pdf_cell … recovered`
  warnings; none of the audited tables were damaged by it.
- JSON is heavy: ~110 KB/page → a full TRM JSON would be ≈ 970 MB. TRM
  likely wants md-only (or chunked JSON); core docs are fine (4–12 MB).
- md is larger than pymupdf's (116 KB vs 64 KB carrier) — docling pads
  table columns with spaces; content-equivalent.

## Recommendation (for the decision gate)

Proceed to the rebuild on a **JSON-as-source-of-truth + md-as-rendering**
substrate, both produced by one docling pass wired into `fetch.sh`:

1. Citations keep the human-facing surface `doc §section (p. N)` — but
   page resolution moves from text anchors to object provenance.
2. The grader, linter, and golden-set ground truth get re-derived against
   the new substrate (their *questions* are converter-independent).
3. TRM: one background grind at rebuild time; md-only, JSON skipped.

Open questions for the gate: TRM JSON strategy (skip vs chunked), where
the escaping normalization lives (grader vs renderer), and CI's story
(docling in CI is heavy — likely CI checks the committed contract while
the operator machine produces the corpus, as today).
