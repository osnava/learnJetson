#!/usr/bin/env bash
# fetch.sh — materialize the pinned hardware-doc corpus as greppable markdown.
#
# Fresh clone? This is step 4 of SETUP.md:
#     agent/hw-docs/fetch.sh [--core|--full]
#       --core (default)  ~9 MB: everything except the two giants
#       --full            adds the 66 MB Orin SoC TRM + 32 MB carrier
#                         reference-design files (schematics)
#
# Downloads are cached in pdf/, converted markdown lands in md/ (both
# gitignored - NVIDIA redistribution terms; INDEX.md is the committed map).
# Run this on the PC you operate the agent from - the Jetson needs nothing:
# the corpus is operator-side knowledge, the box keeps its memory.
# Python deps (one-time): pip install pymupdf4llm openpyxl
# Without them it falls back to poppler pdftotext (no heading structure).
set -euo pipefail
cd "$(dirname "$0")"

TIER=core
case "${1:-}" in
  --full) TIER=full ;;
  --core|"") ;;
  *) echo "usage: $0 [--core|--full]" >&2; exit 2 ;;
esac

PY=python3; command -v python3 >/dev/null 2>&1 || PY=python

# name|url|ext|tier|gated  (versions + verification dates in INDEX.md)
ITEMS=(
  "datasheet|https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_ds|pdf|core|login"
  "orin-nx-nano-design-guide|https://developer.nvidia.com/downloads/jetson-orin-nx-series-nano-series-design-guide|pdf|core|"
  "devkit-carrier-spec|https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf|pdf|core|"
  "orin-pinmux|https://developer.nvidia.com/downloads/jetson-orin-nx-and-orin-nano-series-pinmux-config-template|xlsx|core|"
  "orin-thermal-design-guide|https://developer.nvidia.com/downloads/jetson-orin-nx-orin-nano-series-thermal-design-guide|pdf|core|"
  "orin-pin-function-names|https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nx/docs/jetson_orin_nx_orin_nano_pin_function_names_guide_da-11434-001_v1.0.pdf|pdf|core|"
  "orin-trm|https://developer.nvidia.com/downloads/orin-series-soc-technical-reference-manual/|pdf|full|"
)

mkdir -p pdf md

for item in "${ITEMS[@]}"; do
  IFS='|' read -r name url ext tier gated <<< "$item"
  [ "$tier" = full ] && [ "$TIER" = core ] && continue
  src="pdf/$name.$ext"; out="md/$name.md"

  if [ ! -s "$src" ]; then
    echo ">> downloading $name"
    curl -fsSL --retry 2 -o "$src" "$url" || { echo "   download failed: $name"; rm -f "$src"; continue; }
  fi

  # A login-gated or stale URL hands back HTML instead of the document
  if head -c 5 "$src" | grep -qi '<!doc'; then
    if [ "$gated" = login ]; then
      cat <<EOF
   !! $name is behind an NVIDIA login (free account). One-time manual step:
      1. open in a browser:  $url
         (or Jetson Download Center -> search "$name")
      2. log in and download the PDF
      3. save it as  agent/hw-docs/$src
      4. re-run  ./fetch.sh  - the converter will pick it up
EOF
    else
      echo "   !! $name: got HTML instead of PDF - URL stale? ($url)"
    fi
    rm -f "$src"; continue
  fi

  if [ "$src" -nt "$out" ] || [ ! -s "$out" ]; then
    echo ">> converting $name -> md/"
    "$PY" convert.py "$src" "$out" || { echo "   conversion failed: $name"; rm -f "$out"; continue; }
  else
    echo "   cached: md/$name.md"
  fi
done

if [ "$TIER" = full ]; then
  zip_src="pdf/devkit-carrier-reference-design.zip"
  if [ ! -s "$zip_src" ]; then
    echo ">> downloading devkit-carrier-reference-design (32 MB, schematics)"
    curl -fsSL --retry 2 -o "$zip_src" \
      "https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_reference_design_files_a04_20230320.zip" \
      || { echo "   download failed: reference design"; rm -f "$zip_src"; }
  fi
  if [ -s "$zip_src" ]; then
    mkdir -p pdf/devkit-carrier-reference-design
    unzip -oq "$zip_src" -d pdf/devkit-carrier-reference-design
    echo "   unpacked: pdf/devkit-carrier-reference-design/ (schematics, BOM, gerbers)"

    sch=pdf/devkit-carrier-reference-design/P3768_A04_Concept_schematics.pdf
    sch_out=md/devkit-carrier-schematics.md
    if [ ! -s "$sch" ]; then
      echo "   !! schematics PDF not found in the zip - layout changed?"
    elif [ "$sch" -nt "$sch_out" ] || [ ! -s "$sch_out" ]; then
      echo ">> converting devkit-carrier-schematics -> md/"
      "$PY" convert.py "$sch" "$sch_out" || { echo "   conversion failed: schematics"; rm -f "$sch_out"; }
    else
      echo "   cached: $sch_out"
    fi
  fi
fi

echo
echo "store: $(find md -maxdepth 1 -type f | wc -l) files in agent/hw-docs/md/ (gitignored). Start from INDEX.md."
[ -f md/datasheet.md ] || echo "note: md/datasheet.md still missing - see the login step above."
