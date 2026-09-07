#!/usr/bin/env python3
"""Provenance-carrying semantic search over the v2 corpus (issue #28).

The retrieval step between INDEX routing and declaring a question
not-sourceable. Answers carry heading breadcrumbs and PDF page numbers
resolved from docling object provenance — never guessed.

    python v2/search.py [--index-dir md/index] [--doc STEM] [--k 8] [--json] QUERY

Output (one block per hit):

    devkit-carrier-spec §3.4 Button Header (p. 28)
      | 8 | SYS_RESET* | 239 | Temporarily connect pins 7 and 8 ...

Exit codes mirror grade.py: 0 = hits found, 1 = query given but no index
for the requested doc, 2 = no index at all (corpus not built — run
fetch.sh; not the agent's fault).

Works under any Python >= 3.10: when torch/transformers are missing it
re-execs itself under a docling-capable interpreter ($DOCLING_PY, or the
uv tool env fetch.sh uses).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import embed  # noqa: E402
import normalize  # noqa: E402

SNIPPET = 200


def _docling_python() -> str | None:
    """Locate an interpreter that can embed (torch + transformers)."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModel  # noqa: F401
        return None  # current interpreter works
    except ImportError:
        pass
    env = os.environ.get("DOCLING_PY")
    if env and Path(env).is_file():
        return env
    tool_dir = None
    uv = shutil.which("uv")
    if uv:
        r = subprocess.run([uv, "tool", "dir"], capture_output=True, text=True)
        if r.returncode == 0:
            tool_dir = r.stdout.strip()
    if not tool_dir:
        home = Path.home()
        for cand in (home / "AppData/Roaming/uv/tools",
                     home / ".local/share/uv/tools"):
            if cand.is_dir():
                tool_dir = str(cand)
                break
    if tool_dir:
        for rel in ("Scripts/python.exe", "bin/python"):
            py = Path(tool_dir) / "docling" / rel
            if py.is_file():
                return str(py)
    return None


def _reexec_or_die() -> None:
    py = _docling_python()
    if py is None:
        sys.exit("no torch/transformers under this Python and no docling env "
                 "found — run agent/hw-docs/fetch.sh once, or set DOCLING_PY")
    r = subprocess.run([py, __file__, *sys.argv[1:]])
    sys.exit(r.returncode)


def load_index(index_dir: Path):
    """All shards, with the wrap table applied at load time."""
    import numpy

    wrap_table = normalize.load_wrap_table(index_dir / "wrap_table.json")
    records, vectors = [], []
    for chunks_f in sorted(index_dir.glob("*.chunks.jsonl")):
        shard = chunks_f.name[: -len(".chunks.jsonl")]
        emb_f = index_dir / f"{shard}.emb.npy"
        if not emb_f.is_file():
            continue
        shard_records = [json.loads(line)
                         for line in chunks_f.read_text(encoding="utf-8").splitlines()]
        shard_vec = numpy.load(emb_f)
        if len(shard_records) != len(shard_vec):
            print(f"!! shard {shard}: {len(shard_records)} records vs "
                  f"{len(shard_vec)} vectors — rebuild (fetch.sh)", file=sys.stderr)
            continue
        for r in shard_records:
            r["text"] = normalize.canonical(r["text"], wrap_table)
        records.extend(shard_records)
        vectors.append(shard_vec)
    if not records:
        return None, None
    return records, numpy.vstack(vectors)


def main(argv=None, embed_fn=embed.embed_query) -> int:
    ap = argparse.ArgumentParser(description="semantic search over the v2 corpus")
    ap.add_argument("query", help="natural-language question or identifier")
    ap.add_argument("--index-dir", type=Path, default=Path(__file__).parent.parent / "md/index")
    ap.add_argument("--doc", help="restrict to one corpus stem")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    import numpy  # load/cosine need numpy only; torch just for embedding

    if not args.index_dir.is_dir() or not list(args.index_dir.glob("*.chunks.jsonl")):
        print(f"no index at {args.index_dir} — run agent/hw-docs/fetch.sh first")
        return 2
    records, vectors = load_index(args.index_dir)
    if records is None:
        print(f"index at {args.index_dir} is empty — run fetch.sh again")
        return 2

    if args.doc:
        keep = [i for i, r in enumerate(records) if r["doc"] == args.doc]
        if not keep:
            known = sorted({r["doc"] for r in records})
            print(f"no shard for doc '{args.doc}' (indexed: {', '.join(known)})")
            return 1
        records = [records[i] for i in keep]
        vectors = vectors[keep]

    try:
        q = embed_fn(args.query)
    except ImportError:
        _reexec_or_die()

    scores = vectors @ q
    top = sorted(range(len(records)), key=lambda i: -scores[i])[: args.k]

    hits = []
    for i in top:
        r = records[i]
        section = r["headings"][-1] if r["headings"] else ""
        pages = ",".join(str(p) for p in r["pages"]) or "?"
        hits.append({
            "doc": r["doc"],
            "section": section,
            "breadcrumb": " > ".join(r["headings"]),
            "pages": r["pages"],
            "score": round(float(scores[i]), 4),
            "text": r["text"][:SNIPPET],
        })
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=1))
        return 0
    for h in hits:
        print(f"{h['doc']} §{h['section']} (p. {','.join(str(p) for p in h['pages'])})"
              f"  [{h['score']:.3f}]")
        print(f"  {h['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
