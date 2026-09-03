#!/usr/bin/env python3
"""Convert one fetched hardware document into agent-greppable markdown.

PDF  -> md/<name>.md with a `<!-- p.N -->` anchor before every page, so
        answers can cite "doc, section, p. N" the way INDEX.md asks for.
XLSX -> md/<name>.md (sheet index) + one CSV per sheet beside it — pinmux
        data is tabular and CSV greps better than markdown tables.

Converter choice (2026-09, issue #20): pymupdf4llm — real heading/table
structure on born-digital PDFs, pip-only, no ML models. Marker/Docling/
MinerU only beat it on scanned or complex layouts and need multi-GB model
downloads; NVIDIA's docs are born-digital. Fallback when pymupdf4llm is
absent: poppler `pdftotext -layout` (plain text, headings lost, still
greppable). The corpus lives on the PC the agent operates from — nothing
here ever runs on or lands on the Jetson.
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


def convert_pdf(src: Path, dst: Path) -> None:
    src, dst = src.resolve(), dst.resolve()
    try:
        import pymupdf
        import pymupdf4llm
    except ImportError:
        text = subprocess.run(
            ["pdftotext", "-layout", str(src), "-"],
            check=True, capture_output=True, text=True,
        ).stdout
        pages = text.split("\f")
        body = "\n\n".join(
            f"<!-- p.{i} -->\n\n{p.rstrip()}" for i, p in enumerate(pages, 1) if p.strip()
        )
        dst.write_text(
            f"# {src.stem} (PDF)\n\n"
            "> Converted with the pdftotext fallback - headings are NOT preserved.\n"
            "> `pip install pymupdf4llm` for structured output.\n\n"
            f"{body}\n",
            encoding="utf-8",
        )
        return

    # pymupdf4llm writes images to (and embeds links relative to) the
    # process working directory - so run from the output directory: figures
    # land in md/images/<doc>/ and the inline links resolve relative to the
    # markdown file, for humans in editors and for the agent harness opening
    # them directly (it is multimodal - it reads PNGs itself).
    # Keyed to the OUTPUT stem, not the source: _cull_figures looks figures
    # up by dst.stem, and the two diverge whenever a document is converted
    # under a different name (the carrier schematics) - which silently
    # skipped the cull for exactly that document.
    img_rel = f"images/{dst.stem}"

    # Stream in page batches: bounded memory and visible progress — a
    # 8,800-page TRM does not fit comfortably as one pymupdf4llm result.
    # Image filenames embed the page number, so batches cannot collide.
    BATCH = 200
    n_pages = pymupdf.open(str(src)).page_count
    old_cwd = os.getcwd()
    os.chdir(dst.parent)
    try:
        with dst.open("w", encoding="utf-8", newline="\n") as out:
            out.write(f"# {src.stem} (PDF)\n")
            for start in range(0, n_pages, BATCH):
                batch = list(range(start, min(start + BATCH, n_pages)))
                chunks = pymupdf4llm.to_markdown(
                    str(src), show_progress=False, pages=batch, page_chunks=True,
                    write_images=True, image_path=img_rel,
                )
                if isinstance(chunks, str) or not chunks:
                    # page_chunks=True must yield one dict per page — a bare
                    # string means the API changed; fail loudly instead of
                    # iterating a string character by character.
                    raise RuntimeError(
                        "pymupdf4llm returned %s, expected a list of page dicts"
                        % type(chunks).__name__
                    )
                for offset, chunk in enumerate(chunks):
                    text = chunk.get("text", "").strip()
                    if text:
                        out.write(f"\n\n<!-- p.{start + offset + 1} -->\n\n{text}")
                out.flush()
                if n_pages > BATCH:
                    print(f"   pages {start + 1}-{start + len(batch)}/{n_pages}", flush=True)
    finally:
        os.chdir(old_cwd)
    _cull_figures(dst)


def _figure_ok(path: Path, min_px: int, min_ink: float) -> bool:
    import pymupdf

    try:
        pix = pymupdf.Pixmap(str(path))
        if min(pix.width, pix.height) < min_px:
            return False
        if pix.n > 2 or pix.alpha:
            pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
        s = pix.samples
        step = max(1, len(s) // 50000)
        ink = sum(1 for b in s[::step] if b < 245) / len(s[::step])
        return ink >= min_ink
    except Exception:
        return False


def _cull_figures(dst: Path, min_px: int = 120, min_ink: float = 0.005) -> None:
    """write_images also hoovers up repeated decorations - on the TRM, a
    36 px logo fragment on nearly every page was 97% of all extracted
    images. Drop fragments and near-blank renders; a figure that genuinely
    repeats keeps its link, repointed at the one copy on disk."""
    import hashlib

    img_dir = dst.parent / "images" / dst.stem
    if not img_dir.is_dir():
        return
    kept: dict = {}
    dropped: set = set()
    alias: dict = {}
    for f in sorted(img_dir.glob("*.png")):
        digest = hashlib.md5(f.read_bytes()).hexdigest()
        if digest in kept:
            alias[f.name] = kept[digest]      # same figure, already on disk
        elif _figure_ok(f, min_px, min_ink):
            kept[digest] = f.name
            continue
        dropped.add(f.name)

    if not dropped:
        return

    # One pass over the markdown, matching any figure link and deciding per
    # match - a re.sub per dropped file would rescan the text once each, and
    # on the TRM that is ~8,800 passes over 13 MB.
    def _sub(m):
        prefix, name, nl = m.group(1) or "", m.group(2), m.group(3)
        if name in alias:
            return f"![]({prefix}{alias[name]}){nl}"
        return "" if name in dropped else m.group(0)

    text = re.sub(r"!\[\]\(([^)]*/)?([^)/]+\.png)\)(\n?)", _sub,
                  dst.read_text(encoding="utf-8"))
    dst.write_text(text, encoding="utf-8", newline="\n")
    for name in dropped:
        (img_dir / name).unlink(missing_ok=True)
    print(f"   culled {len(dropped)} fragment/blank/duplicate figures "
          f"({len(alias)} links repointed)", flush=True)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "sheet"


def convert_xlsx(src: Path, dst: Path) -> None:
    import openpyxl

    wb = openpyxl.load_workbook(str(src), read_only=True, data_only=True)
    rows = []
    for ws in wb.worksheets:
        csv_path = dst.parent / f"{src.stem}.{_safe(ws.title)}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in ws.iter_rows(values_only=True):
                if any(c is not None for c in row):
                    writer.writerow(["" if c is None else c for c in row])
        rows.append((ws.title, csv_path.name))
    listing = "\n".join(f"- `{csv}` - sheet '{title}'" for title, csv in rows)
    dst.write_text(
        f"# {src.stem} (XLSX)\n\n"
        "Pinmux workbook - each sheet exported as CSV (one row per pin, grep-friendly):\n\n"
        f"{listing}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <source.pdf|.xlsx> <output.md>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    (convert_xlsx if src.suffix == ".xlsx" else convert_pdf)(src, dst)
    print(f"wrote {dst}")
