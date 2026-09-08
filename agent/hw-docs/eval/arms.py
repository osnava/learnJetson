#!/usr/bin/env python3
"""Build the two cold checkouts the #26 baseline control runs against.

With arm    = the repo as it stands (git export of the measured rev) plus
              the on-disk corpus (md/ with its index shards, pdf/) and the
              gitignored agent/inventory.md — the layer being measured.
Without arm = same export with the layer unwired: the hardware block gone
              from agent/AGENTS.md, INDEX.md present but referenced
              nowhere, and no corpus at all (no md/, no pdf/ — the PDFs
              are corpus content too; leaving them would hand the bare
              model the documents through the Read tool's PDF support).

Both arms additionally lose two trees that are measurement infrastructure
or spike residue, not the layer (answer-key hygiene — a session that finds
the key measures nothing):

  agent/hw-docs/eval/      the golden question set itself
  agent/hw-docs/explore/   the #27 docling spike: a full second copy of the
                           corpus plus FINDINGS.md quoting golden-question
                           material — a retrieval path the layer doesn't own

    python arms.py prepare --repo ROOT --dst SCRATCH [--rev HEAD]
      → SCRATCH/{base,with,without}; exits non-zero if verification fails
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

ROW_OPENER_RE = re.compile(r"^\| `hw-docs/")
READING_ORDER_RE = re.compile(
    r"\s*→\s*\[`hw-docs/INDEX\.md`\]\(hw-docs/INDEX\.md\)"
    r"\s*the moment a question\s+touches hardware\s*→\s*")
README_LINK_RE = re.compile(
    r"\s*·\s*\[Hardware docs index\]\(agent/hw-docs/INDEX\.md\)[^\n]*")
CHECKLIST_ROW_RE = re.compile(r"^\| ☐ \| Hardware corpus fetched.*\n?",
                              re.MULTILINE)


def _drop_tooling_rows(text: str, keep_row) -> str:
    r"""Remove markdown table rows whose first cell is `hw-docs/…`. Rows
    physically wrap in this file (the | `hw-docs/ opener can sit alone
    on its own line), so consume until the line that closes the row."""
    lines = text.splitlines(keepends=True)
    out, i = [], 0
    while i < len(lines):
        if ROW_OPENER_RE.match(lines[i]):
            row = [lines[i]]
            i += 1
            while i < len(lines) and not row[-1].rstrip().endswith("|"):
                row.append(lines[i])
                i += 1
            if keep_row("".join(row)):
                out.extend(row)
        else:
            out.append(lines[i])
            i += 1
    return "".join(out)


def _drop_section(text: str, heading_prefix: str) -> str:
    """Remove a `## heading` section up to (not including) the next `## `."""
    lines = text.splitlines(keepends=True)
    out, skipping = [], False
    for line in lines:
        if skipping:
            if line.startswith("## "):
                skipping = False
            else:
                continue
        elif line.startswith(heading_prefix):
            skipping = True
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "".join(out))


def strip_agents_md(text: str, without: bool) -> str:
    if without:
        text = READING_ORDER_RE.sub(" → ", text)
        text = _drop_section(text, "## Hardware questions")
        text = _drop_tooling_rows(text, keep_row=lambda row: False)
    # both arms: any tooling row pointing into eval/ goes (the golden set
    # is the answer key; a pointer to it from a measured arm measures nothing)
    return _drop_tooling_rows(
        text, keep_row=lambda row: "eval/" not in row)


def strip_readme(text: str) -> str:
    return README_LINK_RE.sub("", text)


def strip_setup(text: str) -> str:
    lines = text.splitlines(keepends=True)
    start = end = None
    for i, line in enumerate(lines):
        if "uv tool install docling" in line:
            j = i
            while j > 0 and lines[j].strip() != "```bash":
                j -= 1
            start = j if lines[j].strip() == "```bash" else i
        if "page provenance." in line:
            end = i
            break
    if start is not None and end is not None and end >= start:
        del lines[start:end + 1]
    text = "".join(lines)
    return CHECKLIST_ROW_RE.sub("", text)


KIT_DOCS = ("agent/AGENTS.md", "README.md", "agent/SETUP.md")


def build_without(src: Path, dst: Path, kit_from: Path | None = None) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)
    (dst / "agent" / "AGENTS.md").write_text(
        strip_agents_md((dst / "agent" / "AGENTS.md").read_text(
            encoding="utf-8"), without=True), encoding="utf-8")
    (dst / "README.md").write_text(
        strip_readme((dst / "README.md").read_text(encoding="utf-8")),
        encoding="utf-8")
    (dst / "agent" / "SETUP.md").write_text(
        strip_setup((dst / "agent" / "SETUP.md").read_text(encoding="utf-8")),
        encoding="utf-8")
    hw = dst / "agent" / "hw-docs"
    if hw.is_dir():  # INDEX.md stays — present, referenced nowhere
        for p in hw.iterdir():
            if p.name == "INDEX.md":
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    if kit_from:
        _copy_kit(kit_from, dst)


def build_with(src: Path, dst: Path, corpus_from: Path | None = None,
               kit_from: Path | None = None) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)
    shutil.rmtree(dst / "agent" / "hw-docs" / "eval", ignore_errors=True)
    shutil.rmtree(dst / "agent" / "hw-docs" / "explore", ignore_errors=True)
    (dst / "agent" / "AGENTS.md").write_text(
        strip_agents_md((dst / "agent" / "AGENTS.md").read_text(
            encoding="utf-8"), without=False), encoding="utf-8")
    if corpus_from:
        for name in ("md", "pdf"):
            if (corpus_from / name).is_dir():
                shutil.copytree(corpus_from / name,
                                dst / "agent" / "hw-docs" / name,
                                dirs_exist_ok=True)
    if kit_from:
        _copy_kit(kit_from, dst)


def _copy_kit(agent_dir: Path, dst: Path) -> None:
    inv = agent_dir / "inventory.md"
    if inv.is_file():
        shutil.copy2(inv, dst / "agent" / "inventory.md")


def verify_without(dst: Path) -> list[str]:
    problems = []
    for f in KIT_DOCS:
        text = (dst / f).read_text(encoding="utf-8")
        if "hw-docs" in text:
            problems.append(f"{f} still references hw-docs")
    hw = dst / "agent" / "hw-docs"
    if not (hw / "INDEX.md").is_file():
        problems.append("INDEX.md missing (issue #26: present, unreferenced)")
    extra = [p.name for p in hw.iterdir() if p.name != "INDEX.md"]
    if extra:
        problems.append(f"hw-docs has more than INDEX.md: {extra}")
    return problems


def verify_with(dst: Path) -> list[str]:
    problems = []
    hw = dst / "agent" / "hw-docs"
    for needed in ("INDEX.md", "search.py", "md"):
        if not (hw / needed).exists():
            problems.append(f"with arm missing {needed}")
    for banned in ("eval", "explore"):
        if (hw / banned).exists():
            problems.append(f"with arm still carries {banned}/ (answer-key hygiene)")
    agents_md = (dst / "agent" / "AGENTS.md").read_text(encoding="utf-8")
    if "eval/questions.yaml" in agents_md:
        problems.append("AGENTS.md still points at the golden set")
    if "hw-docs/INDEX.md" not in agents_md:
        problems.append("with arm lost its INDEX.md routing reference")
    return problems


def export_repo(repo: Path, rev: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", rev],
                         capture_output=True)
    if tar.returncode != 0:
        sys.exit(f"git archive failed: {tar.stderr.decode(errors='replace')}")
    untar = subprocess.run(["tar", "-xf", "-", "-C", str(dst)],
                           input=tar.stdout, capture_output=True)
    if untar.returncode != 0:
        sys.exit(f"tar extract failed: {untar.stderr.decode(errors='replace')}")


def prepare(repo: Path, dst: Path, rev: str) -> dict[str, list[str]]:
    base = dst / "base"
    export_repo(repo, rev, base)
    build_with(base, dst / "with", corpus_from=repo / "agent" / "hw-docs",
               kit_from=repo / "agent")
    build_without(base, dst / "without", kit_from=repo / "agent")
    return {"with": verify_with(dst / "with"),
            "without": verify_without(dst / "without")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="export + build both arms + verify")
    p.add_argument("--repo", type=Path, default=HERE.parents[2])
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--rev", default="HEAD")
    args = ap.parse_args(argv)

    problems = prepare(args.repo, args.dst, args.rev)
    for arm, probs in problems.items():
        for prob in probs:
            print(f"VERIFY FAIL [{arm}]: {prob}", file=sys.stderr)
    if any(problems.values()):
        return 1
    print(f"arms ready under {args.dst} (with, without)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
