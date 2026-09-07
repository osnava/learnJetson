#!/usr/bin/env bash
# fetch.sh — materialize the pinned hardware-doc corpus (v2, issue #28).
#
# Fresh clone? This is step 4 of SETUP.md:
#     agent/hw-docs/fetch.sh [--core|--full]
#       --core (default)  ~9 MB: everything except the two giants
#       --full            adds the 66 MB Orin SoC TRM + 32 MB carrier
#                         reference-design files (schematics)
#
# Downloads are cached in pdf/. The v2 substrate (docling, #27/#28):
# JSON is the source of truth (md/<doc>.json for the five core docs),
# md/<doc>.md is the normalized rendering agents read, and md/index/
# holds the semantic-search shards (chunks + embeddings + wrap table).
# All gitignored - NVIDIA redistribution terms; INDEX.md is the map.
# Run this on the PC you operate the agent from - the Jetson needs nothing:
# the corpus is operator-side knowledge, the box keeps its memory.
# One-time deps: a docling-capable Python (uv tool install docling) -
# resolved automatically below. GPU is used when present (≈6.5 min for
# the core corpus, measured); the TRM is a ~7.3 h background grind.
set -euo pipefail
cd "$(dirname "$0")"

TIER=core
case "${1:-}" in
  --full) TIER=full ;;
  --core|"") ;;
  *) echo "usage: $0 [--core|--full]" >&2; exit 2 ;;
esac

PY=python3; command -v python3 >/dev/null 2>&1 || PY=python

# --- resolve a docling-capable Python (v2 conversion substrate) --------
resolve_docling_py() {
  if [ -n "${DOCLING_PY:-}" ] && [ -x "${DOCLING_PY%% *}" ]; then
    echo "$DOCLING_PY"; return 0
  fi
  if "$PY" -c "import docling" >/dev/null 2>&1; then
    echo "$PY"; return 0
  fi
  local dir=""
  if command -v uv >/dev/null 2>&1; then
    dir="$(uv tool dir 2>/dev/null || true)"
  fi
  local cand
  for cand in \
    "${dir:-/nonexistent}/docling/Scripts/python.exe" \
    "${dir:-/nonexistent}/docling/bin/python" \
    "$HOME/AppData/Roaming/uv/tools/docling/Scripts/python.exe" \
    "$HOME/.local/share/uv/tools/docling/bin/python"
  do
    if [ -f "$cand" ]; then echo "$cand"; return 0; fi
  done
  return 1
}

if ! DOCLING_PY="$(resolve_docling_py)"; then
  cat >&2 <<'EOF'
 !! No docling-capable Python found. The v2 corpus needs docling:
      uv tool install docling     (recommended; pulls GPU torch)
      # or: pip install docling
    Models download on first conversion (cached user-globally).
    Then re-run ./fetch.sh
EOF
  exit 1
fi

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

mkdir -p pdf md/index

# stale = source newer than a required artifact, or that artifact missing.
# Args: src out [more required artifacts...] — any missing/newer -> stale.
is_stale() {
  local src=$1 art
  shift
  for art in "$@"; do
    [ "$src" -nt "$art" ] && return 0
    [ -s "$art" ] || return 0
  done
  return 1
}

# stale = source newer than rendering, rendering missing, or search
# shard missing. Stale sources go to ONE build.py call so the wrap
# table derives across documents (a wrapped cell in the carrier spec is
# confirmed by the pin-name spelled whole in the function-names guide).
BUILD_ARGS=()
stale=0

for item in "${ITEMS[@]}"; do
  IFS='|' read -r name url ext tier gated <<< "$item"
  # full-tier giants are downloaded + handled by the --full section
  # below (the TRM as a detached slab grind) - never by this batch call
  [ "$tier" = full ] && continue
  src="pdf/$name.$ext"; out="md/$name.md"; shard="md/index/$name.chunks.jsonl"

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

  if [ "$ext" = xlsx ]; then
    # pinmux: xlsx -> md + per-sheet CSVs (path unchanged from v1)
    if is_stale "$src" "$out"; then
      BUILD_ARGS+=("$src"); stale=1
    else
      echo "   cached: $out"
    fi
  elif is_stale "$src" "$out" "$shard" "md/$name.json"; then
    # json is the source of truth - a missing one means rebuild too
    BUILD_ARGS+=("$src"); stale=1
  else
    echo "   cached: $out (+ json + index shard)"
  fi
done

if [ "$stale" = 1 ]; then
  echo ">> converting (docling: json source of truth + md rendering + search index)"
  if ! "$DOCLING_PY" v2/build.py --md-dir md --index-dir md/index \
       --persist-json "${BUILD_ARGS[@]}"; then
    echo "   !! conversion reported failures - see output above"
  fi
else
  echo ">> core corpus up to date (json + md + index)"
fi

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
    sch_shard=md/index/devkit-carrier-schematics.chunks.jsonl
    if [ ! -s "$sch" ]; then
      echo "   !! schematics PDF not found in the zip - layout changed?"
    elif is_stale "$sch" "$sch_out" "$sch_shard"; then
      echo ">> converting devkit-carrier-schematics -> md/ + index"
      "$DOCLING_PY" v2/build.py --md-dir md --index-dir md/index \
        "$sch=devkit-carrier-schematics" \
        || { echo "   conversion failed: schematics"; }
    else
      echo "   cached: $sch_out (+ index shard)"
    fi
  fi

  # TRM: md-only (a full JSON would be ~970 MB), slab-wise for bounded
  # memory and resumability. ~7.3 h measured extrapolation on GPU - run
  # detached, tail the log; each 250-page slab's shard is atomic, so a
  # crashed or interrupted grind resumes where it stopped.
  trm=pdf/orin-trm.pdf; trm_out=md/orin-trm.md
  trm_url="https://developer.nvidia.com/downloads/orin-series-soc-technical-reference-manual/"
  if [ ! -s "$trm" ]; then
    echo ">> downloading orin-trm (66 MB)"
    curl -fsSL --retry 2 -o "$trm" "$trm_url" \
      || { echo "   download failed: orin-trm"; rm -f "$trm"; }
  fi
  # Complete = md written (build.py only assembles it from a full slab
  # set) AND no leftover slab parts AND at least one v2 shard (a v1-era
  # md alone is not a completion certificate). Leftover parts mean an
  # interrupted grind - resume, don't call it cached.
  trm_parts=$(compgen -G "md/orin-trm.p*.raw.md" 2>/dev/null || true)
  trm_shards=$(compgen -G "md/index/orin-trm.p*.chunks.jsonl" 2>/dev/null || true)
  if [ ! -s "$trm" ]; then
    echo "   !! $trm not cached - download failed above?"
  elif head -c 5 "$trm" | grep -qi '<!doc'; then
    echo "   !! orin-trm: got HTML instead of PDF - URL stale? ($trm_url)"
    rm -f "$trm"
  elif is_stale "$trm" "$trm_out" || [ ! -s "$trm_out" ] \
       || [ -n "$trm_parts" ] || [ -z "$trm_shards" ]; then
    log=md/index/orin-trm.build.log
    echo ">> orin-trm: background grind starting (~7.3 h, md-only, resumable)"
    if command -v nohup >/dev/null 2>&1; then
      nohup "$DOCLING_PY" v2/build.py --md-dir md --index-dir md/index \
        --slab-pages 250 "$trm" > "$log" 2>&1 &
    else
      "$DOCLING_PY" v2/build.py --md-dir md --index-dir md/index \
        --slab-pages 250 "$trm" > "$log" 2>&1 &
    fi
    echo "   log: $log    (tail -f agent/hw-docs/$log; re-run fetch.sh to resume)"
  else
    echo "   cached: $trm_out (+ index shards)"
  fi
fi

echo
echo "store: $(find md -maxdepth 1 -type f | wc -l) files in agent/hw-docs/md/ (gitignored)."
echo "search: python v2/search.py \"question\"   (INDEX routes first, search when routing misses)"
[ -f md/datasheet.md ] || echo "note: md/datasheet.md still missing - see the login step above."
