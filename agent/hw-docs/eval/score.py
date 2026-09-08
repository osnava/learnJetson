#!/usr/bin/env python3
"""Cold-session scoring: transcripts → per-item signals → scoreboard → delta.

Issues #25 (runner) and #26 (baseline control). score.py is the read side
of the harness: it parses `claude -p --output-format stream-json`
transcripts produced by run.sh, extracts the six per-item signals, grades
citations through the structural grader (../grade.py — JSON provenance,
same grader for BOTH arms, always against the real corpus so a bare
model's recited pages are checked against reality), optionally runs an
LLM judge for answer correctness, and renders the per-category scoreboard
plus the with-vs-without delta table.

Per-item signals (#25 acceptance):
  routed        hw-docs/INDEX.md was opened before the answer
  searched      search.py was invoked (the v2 route → search → declare order)
  cited         the answer carries ≥1 gradeable `doc §section (p. N)`
  grade_exit    grade.py over the answer: 0 valid / 1 invalid / 2 unverifiable
  verdict       judge: CORRECT|PARTIAL|WRONG (answerable) or
                ABSTAINED|PARTIAL|GUESSED (unanswerable)
  forum_only    the answer rests solely on a forum/blog
  contaminated  the session read eval/questions.yaml — the answer key.
                Contaminated rows are scored but flagged; a clean run has 0.

    python score.py SCORE_DIR --questions eval/questions.yaml \
        --corpus ../md --out results/<label> [--no-judge]

SCORE_DIR layout (written by run.sh):
    transcripts/<arm>/<id>.ndjson     raw session transcripts
    meta/<arm>/<id>.json              {duration_s, retries, arm, id}

Output: answers/<arm>/<id>.md, judge/<arm>/<id>.json, <arm>.items.json,
scoreboard-<arm>.md, delta.md (when both arms present), summary.json,
transcripts/<arm>/ (committed copy: structure kept, tool-result bodies
truncated) and raw/<arm>/ (verbatim transcripts — operator-side like the
corpus, gitignored).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # grade.py lives flat in hw-docs/
import grade  # noqa: E402

# matched against a json.dumps blob, where a backslash appears doubled
INDEX_RE = re.compile(r"hw-docs[/\\]+INDEX\.md", re.IGNORECASE)
SEARCH_RE = re.compile(r"search\.py", re.IGNORECASE)
KEY_RE = re.compile(r"eval[/\\]questions\.ya?ml", re.IGNORECASE)

VERDICTS_ANSWERABLE = {"CORRECT", "PARTIAL", "WRONG"}
VERDICTS_UNANSWERABLE = {"ABSTAINED", "PARTIAL", "GUESSED"}


# --- transcript parsing ------------------------------------------------------

def parse_transcript(text: str) -> dict:
    """NDJSON stream-json → metadata, ordered tool uses, final answer."""
    out = {"model": None, "session_id": None, "num_turns": None,
           "cost_usd": None, "final_answer": None, "tool_uses": [],
           "ok": False, "error": None}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue  # progress noise / torn line — the result event rules
        et = ev.get("type")
        if et == "system" and ev.get("subtype") == "init":
            out["model"] = ev.get("model")
            out["session_id"] = ev.get("session_id")
        elif et == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if block.get("type") == "tool_use":
                    out["tool_uses"].append(
                        {"tool": block.get("name"), "input": block.get("input") or {}})
        elif et == "result":
            out["final_answer"] = ev.get("result")
            out["num_turns"] = ev.get("num_turns")
            out["cost_usd"] = ev.get("total_cost_usd")
            if ev.get("is_error"):
                out["error"] = ev.get("result") or "session ended with is_error"
            else:
                out["ok"] = True
    if out["final_answer"] is None and out["error"] is None:
        out["error"] = "no result event in transcript"
    return out


def routing_signals(tool_uses: list[dict]) -> dict:
    """Transcript-side signals, from what the session actually did."""
    sig = {"routed": False, "searched": False, "contaminated": False}
    for use in tool_uses:
        blob = json.dumps(use["input"])
        if KEY_RE.search(blob):
            sig["contaminated"] = True
        if INDEX_RE.search(blob):
            sig["routed"] = True
        if SEARCH_RE.search(blob):
            sig["searched"] = True
    return sig


# --- citation signals (same grader, same corpus, both arms) -------------------

def citation_signals(answer: str, corpus_dir: Path) -> dict:
    """cited + grade.py exit over the answer text, via ../grade.py."""
    citations = grade.parse_answer(answer or "")
    if not citations:
        return {"cited": False, "grade_exit": None}
    results = grade.grade(citations, Path(corpus_dir))
    return {"cited": True, "grade_exit": grade.exit_code(results)}


# --- judge --------------------------------------------------------------------

def judge_prompt(item: dict, answer: str) -> str:
    answerable = item.get("answerable")
    ground = item.get("expected_answer", "").strip()
    if answerable:
        rubric = """Verdict — exactly one of:
  CORRECT  every load-bearing fact of the ground truth is present and asserted
           correctly (numbers, names, yes/no, which domain lives where); no
           contradicting claim.
  PARTIAL  right direction, but one load-bearing fact is missing or wrong.
  WRONG    asserts a conclusion that contradicts the ground truth, or recites
           fabricated specifics.
forum_only: true iff the answer's sole support is a forum/blog post.
Judge SUBSTANCE only — ignore citation format, sections and page numbers;
those are graded by a separate machine check."""
    else:
        rubric = """Verdict — exactly one of:
  ABSTAINED  explicitly declines to answer from the local documents ("not
             sourceable locally" or equivalent) AND hands back a concrete way
             to check (a command or a named official source, matching the
             spirit of the expected redirect).
  PARTIAL    declines but offers no real way to check.
  GUESSED    gives a substantive answer (a number, a procedure) as fact.
forum_only: true iff forum/blog content is presented as the answer."""
    return f"""You are grading one hardware Q&A answer against established ground truth.
Reply with ONE line of JSON and nothing else:
{{"verdict": "...", "forum_only": true|false, "note": "<= 20 words"}}

{rubric}

QUESTION: {item['question']}

GROUND TRUTH{'' if answerable else ' (the correct behaviour is abstention + redirect)'}:
{ground}
{chr(10) + 'EXPECTED REDIRECT:' + chr(10) + item['expected_redirect'].strip() if not answerable else ''}

ANSWER TO GRADE:
{answer}"""


def parse_judge(text: str) -> dict:
    """First JSON object in the judge reply → verdict dict; garbage → PARSE_ERROR."""
    for m in re.finditer(r"\{[^{}]*\}", text or "", re.DOTALL):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(d.get("verdict"), str):
            v = d["verdict"].strip().upper()
            if v in VERDICTS_ANSWERABLE | VERDICTS_UNANSWERABLE:
                return {"verdict": v,
                        "forum_only": bool(d.get("forum_only", False)),
                        "note": str(d.get("note", ""))[:200]}
    return {"verdict": "PARSE_ERROR", "forum_only": False, "note": (text or "")[:200]}


def run_judge(prompt: str, cmd: list[str], retries: int = 1,
              timeout: int = 180) -> dict:
    """claude -p with the prompt on stdin; returns judge dict + raw output."""
    last = ""
    for _ in range(retries + 1):
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True,
                               text=True, timeout=timeout, encoding="utf-8")
        except (subprocess.TimeoutExpired, OSError) as e:
            last = f"judge subprocess failed: {e}"
            time.sleep(2)
            continue
        last = r.stdout
        try:
            result = json.loads(r.stdout).get("result", "")
        except ValueError:
            result = ""
        judged = parse_judge(result or r.stdout)
        judged["raw"] = result or r.stdout
        if judged["verdict"] != "PARSE_ERROR":
            return judged
    return {"verdict": "PARSE_ERROR", "forum_only": False,
            "note": "judge unparseable", "raw": last}


# --- scoring ------------------------------------------------------------------

def item_grouping(item: dict) -> str:
    if not item.get("answerable"):
        return "unanswerable"
    return "divergent" if item.get("divergence") else "already-knows"


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def scoreboard(rows: list[dict]) -> dict:
    """rows → {slice_name: metric dict}. Slices: all, each group, each category."""
    slices: dict[str, list[dict]] = {"all": rows}
    for key in ("divergent", "already-knows", "unanswerable"):
        slices[key] = [r for r in rows if r["group"] == key]
    for cat in sorted({r["category"] for r in rows}):
        slices[cat] = [r for r in rows if r["category"] == cat]

    out = {}
    for name, items in slices.items():
        n = len(items)
        ans = [r for r in items if r["answerable"]]
        unans = [r for r in items if not r["answerable"]]

        def cnt(pred, pool=None):
            return sum(1 for r in (pool if pool is not None else items) if pred(r))

        m = {
            "n": n,
            "errors": cnt(lambda r: not r["ok"]),
            "unscored": cnt(lambda r: r["ok"] and r["verdict"] is None),
            "routed": _rate(cnt(lambda r: r["routed"]), n),
            "searched": _rate(cnt(lambda r: r["searched"]), n),
            "cited": _rate(cnt(lambda r: r["cited"]), n),
            "valid": _rate(cnt(lambda r: r["grade_exit"] == 0), n),
            "invalid": _rate(cnt(lambda r: r["grade_exit"] == 1), n),
            "unverifiable": _rate(cnt(lambda r: r["grade_exit"] == 2), n),
            "forum_only": cnt(lambda r: r["forum_only"]),
            "contaminated": cnt(lambda r: r["contaminated"]),
            "cost_usd": round(sum(r["cost_usd"] or 0 for r in items), 2),
        }
        if ans:
            m.update({
                "correct": _rate(cnt(lambda r: r["verdict"] == "CORRECT", ans), len(ans)),
                "partial": _rate(cnt(lambda r: r["verdict"] == "PARTIAL", ans), len(ans)),
                "wrong": _rate(cnt(lambda r: r["verdict"] == "WRONG", ans), len(ans)),
                "answerable_n": len(ans),
            })
        if unans:
            m.update({
                "abstained": _rate(cnt(lambda r: r["verdict"] == "ABSTAINED", unans), len(unans)),
                "guessed": _rate(cnt(lambda r: r["verdict"] == "GUESSED", unans), len(unans)),
                "unanswerable_n": len(unans),
            })
        out[name] = m
    return out


_DELTA_ROWS = [  # (metric key, header, applies to)
    ("routed", "routed", "all"),
    ("searched", "searched", "all"),
    ("cited", "cited %", "all"),
    ("valid", "citations valid %", "all"),
    ("invalid", "citations invalid %", "all"),
    ("correct", "correct %", "answerable"),
    ("abstained", "abstained correctly %", "unanswerable"),
    ("guessed", "guessed %", "unanswerable"),
    ("forum_only", "forum-only (count)", "all"),
    ("unscored", "unscored (count)", "all"),
    ("contaminated", "contaminated (count)", "all"),
    ("errors", "session errors", "all"),
    ("cost_usd", "cost $", "all"),
]


def _fmt(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}"


def _delta(a, b) -> str:
    if a is None or b is None:
        return "—"
    d = 100 * (a - b)
    return "±0.0" if abs(d) < 0.05 else f"{d:+.1f}"


def render_delta(with_rows: list[dict], without_rows: list[dict]) -> str:
    """The #26 deliverable: per-metric, per-slice with-vs-without delta."""
    sb_with = scoreboard(with_rows)
    sb_without = scoreboard(without_rows)
    lines = ["# Baseline delta — with vs without the knowledge layer", ""]
    order = ["divergent", "already-knows", "unanswerable", "all"] + [
        s for s in sb_with if s not in
        ("divergent", "already-knows", "unanswerable", "all")]
    for slice_name in order:
        if slice_name not in sb_with:
            continue
        w, wo = sb_with[slice_name], sb_without.get(slice_name)
        lines += [f"## {slice_name} (n={w['n']})", "",
                  "| metric | with | without | delta |", "|---|---|---|---|"]
        for key, header, applies in _DELTA_ROWS:
            if applies == "answerable" and "answerable_n" not in w:
                continue
            if applies == "unanswerable" and "unanswerable_n" not in w:
                continue
            if key not in w:
                continue
            raw = w[key]
            if key == "cost_usd":
                fmt = lambda v: "—" if v is None else f"{v:.2f}"
                d = ("—" if raw is None or wo is None or key not in wo
                     else f"{raw - wo[key]:+.2f}")
            elif key in ("forum_only", "errors", "unscored", "contaminated"):
                fmt = lambda v: "—" if v is None else str(v)
                d = ("—" if raw is None or wo is None or key not in wo
                     else f"{raw - wo[key]:+g}")
            else:
                fmt = _fmt
                d = _delta(raw, wo.get(key))
            lines.append(f"| {header} | {fmt(raw)} | "
                         f"{fmt(wo.get(key)) if wo else '—'} | {d} |")
        lines.append("")
    return "\n".join(lines)


def render_scoreboard(arm: str, rows: list[dict]) -> str:
    sb = scoreboard(rows)
    lines = [f"# Scoreboard — {arm} arm", ""]
    for name, m in sb.items():
        lines.append(f"## {name} (n={m['n']})")
        bits = []
        for key in ("routed", "searched", "cited", "valid", "invalid",
                    "unverifiable", "correct", "abstained", "guessed"):
            if key in m:
                bits.append(f"{key} {_fmt(m[key])}%")
        bits += [f"forum-only {m['forum_only']}", f"errors {m['errors']}",
                 f"unscored {m['unscored']}", f"contaminated {m['contaminated']}",
                 f"cost ${m['cost_usd']}"]
        lines.append("  · ".join(bits) + "\n")
    return "\n".join(lines)


# --- transcript slimming for the committed copy -------------------------------

def filter_transcript(text: str, keep: int = 500) -> str:
    """Keep structure, truncate tool_result bodies — the committed copy stays
    readable (what was read / greped / run) without the megabytes of corpus
    text the tool results carry."""
    out_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            out_lines.append(line)
            continue
        content = (ev.get("message") or {}).get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result" \
                        and isinstance(block.get("content"), str) \
                        and len(block["content"]) > keep:
                    block["content"] = block["content"][:keep] + "…"
        out_lines.append(json.dumps(ev, ensure_ascii=False))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


# --- CLI -----------------------------------------------------------------------

def _questions_path() -> Path:
    return HERE / "questions.yaml"


def _load_questions(path: Path) -> list[dict]:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def score_arm(arm: str, questions: list[dict], score_dir: Path, out_dir: Path,
              corpus: Path, judge_cmd: list[str] | None) -> list[dict]:
    rows = []
    tdir = score_dir / "transcripts" / arm
    mdir = score_dir / "meta" / arm
    (out_dir / "answers" / arm).mkdir(parents=True, exist_ok=True)
    (out_dir / "judge" / arm).mkdir(parents=True, exist_ok=True)
    for q in questions:
        row = {"id": q["id"], "category": q["category"],
               "answerable": q["answerable"], "group": item_grouping(q),
               "routed": False, "searched": False, "contaminated": False,
               "cited": False, "grade_exit": None, "verdict": None,
               "forum_only": False, "ok": False, "cost_usd": None,
               "num_turns": None, "duration_s": None, "error": None}
        tf = tdir / f"{q['id']}.ndjson"
        meta_f = mdir / f"{q['id']}.json"
        if meta_f.is_file():
            try:
                meta = json.loads(meta_f.read_text(encoding="utf-8"))
                row["duration_s"] = meta.get("duration_s")
                row["retries"] = meta.get("retries", 0)
            except ValueError:
                pass
        if not tf.is_file():
            row["error"] = "transcript missing"
            rows.append(row)
            continue
        p = parse_transcript(tf.read_text(encoding="utf-8"))
        row.update({k: p[k] for k in
                    ("model", "session_id", "num_turns", "cost_usd", "ok", "error")})
        row.update(routing_signals(p["tool_uses"]))
        answer = p["final_answer"] or ""
        (out_dir / "answers" / arm / f"{q['id']}.md").write_text(
            answer, encoding="utf-8")
        if p["ok"]:
            row.update(citation_signals(answer, corpus))
            if judge_cmd:
                judged = run_judge(judge_prompt(q, answer), judge_cmd)
                row["verdict"] = judged["verdict"]
                row["forum_only"] = judged["forum_only"]
                (out_dir / "judge" / arm / f"{q['id']}.json").write_text(
                    json.dumps(judged, ensure_ascii=False, indent=1),
                    encoding="utf-8")
        rows.append(row)
        print(f"  [{arm}] {q['id']}: ok={row['ok']} verdict={row['verdict']} "
              f"routed={row['routed']} cited={row['cited']} "
              f"grade_exit={row['grade_exit']}")
    slim = out_dir / "transcripts" / arm
    raw = out_dir / "raw" / arm
    slim.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    for q in questions:
        tf = tdir / f"{q['id']}.ndjson"
        if tf.is_file():
            shutil.copyfile(tf, raw / f"{q['id']}.ndjson")  # full, post-hoc
            (slim / f"{q['id']}.ndjson").write_text(
                filter_transcript(tf.read_text(encoding="utf-8")),
                encoding="utf-8")
    (out_dir / f"{arm}.items.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"scoreboard-{arm}.md").write_text(
        render_scoreboard(arm, rows), encoding="utf-8")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("score_dir", type=Path,
                    help="run dir with transcripts/<arm>/<id>.ndjson")
    ap.add_argument("--questions", type=Path, default=_questions_path())
    ap.add_argument("--corpus", type=Path, required=True,
                    help="md/ corpus both arms are graded against (the real one)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-judge", action="store_true",
                    help="skip LLM judging (signals + citations only)")
    ap.add_argument("--judge-cmd", default="claude -p --output-format json",
                    help="judge invocation (prompt arrives on stdin)")
    args = ap.parse_args(argv)

    questions = _load_questions(args.questions)
    args.out.mkdir(parents=True, exist_ok=True)
    arms = sorted(p.name for p in (args.score_dir / "transcripts").iterdir()
                  if p.is_dir())
    if not arms:
        sys.exit(f"no transcripts/<arm>/ under {args.score_dir}")
    judge_cmd = None if args.no_judge else args.judge_cmd.split()

    arm_rows = {}
    for arm in arms:
        print(f"scoring arm: {arm}")
        arm_rows[arm] = score_arm(arm, questions, args.score_dir, args.out,
                                  args.corpus, judge_cmd)

    summary = {"scored_at": _dt.datetime.now().isoformat(timespec="seconds"),
               "questions": len(questions), "arms": {},
               "judge": "none" if args.no_judge else " ".join(judge_cmd)}
    for arm, rows in arm_rows.items():
        ok_rows = [r for r in rows if r["ok"]]
        summary["arms"][arm] = {
            "models": sorted({r.get("model") for r in ok_rows if r.get("model")}),
            "cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 2),
            "turns_total": sum(r["num_turns"] or 0 for r in rows),
            "errors": sum(1 for r in rows if not r["ok"]),
            "contaminated": sum(1 for r in rows if r["contaminated"]),
        }
    if "with" in arm_rows and "without" in arm_rows:
        (args.out / "delta.md").write_text(
            render_delta(arm_rows["with"], arm_rows["without"]),
            encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
