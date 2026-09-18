#!/usr/bin/env python3
r"""
bundle.py - Turn a flat edgar_fetch corpus into one upload per template stage.

Each bundle is a single markdown file holding: the instructions for that stage,
the template sections it must write, and the source filings it may use, each
headed with its accession number so citations stay traceable.

Run order:
    1. Bundles A, B, C, D2  (independent; any order)
    2. Save each reviewed reply as drafts/A.md, drafts/B.md, ...
    3. Bundle D (reads drafts/D2.md)
    4. Fill in market_data.md, then bundle E
    5. Bundle F (enable web search in that chat), then bundle G

Usage (PowerShell):
    python bundle.py --corpus C:\Users\markc\Downloads\corpus-flat\APLD --template blog_post_template_V3.md
    python bundle.py --corpus ... --template ... --only E      # rebuild one bundle after drafts exist
    python bundle.py --corpus ... --template ... --check       # report sizes only, write nothing
    python bundle.py --corpus ... --template ... --check --detail --only D2   # every source in one bundle

Requires the corpus to have been fetched with --flat (it reads manifest.json).
Use --split too, or every 10-K/10-Q selector will come back empty.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

# Markdown tables tokenize worse than prose; 3.5 chars/token errs high on purpose.
CHARS_PER_TOKEN = 3.5

SPLIT_RE = re.compile(r"_item(\d+[A-Z]?)(-\d+)?(?:-[^.]*)?\.md$")
ITEM_HEADING_RE = re.compile(r"^## ", re.MULTILINE)

MARKET_DATA_TEMPLATE = """# Market data - {ticker}
*Enter by hand from EODHD / Morningstar. Every field you fill here is part of
the spec for the automated feed later. Leave N/A where a value doesn't exist.*

As of (date/time):
Source(s):

## Snapshot
| Field | Value | Source |
|---|---|---|
| Price | | |
| 52-week high / low | | |
| YTD change % | | |
| 1-day / 1-week change % (if recent catalyst) | | |
| Shares outstanding used (from drafts/D.md, all classes) | | |
| Market cap | | |
| Total debt (incl. convertibles) | | |
| Cash & equivalents | | |
| Operating/finance lease liabilities (include in EV? y/n) | | |
| Enterprise value | | |

## Multiples
| Metric | Current | Own 1yr low-high (median) | Own 3yr low-high (median) |
|---|---|---|---|
| Trailing P/E | | | |
| Forward P/E | | | |
| EV/Sales (fwd) | | | |
| EV/EBITDA (fwd) | | | |
| P/FCF | | | |

## Peer set (name it and state its purpose)
Purpose:
| Peer | EV/Sales fwd | EV/EBITDA fwd | Fwd P/E | Gross margin | Op margin |
|---|---|---|---|---|---|
| | | | | | |

## Consensus (label as analyst estimates)
| Metric | Next FY | FY+1 | # analysts | Source |
|---|---|---|---|---|
| Revenue | | | | |
| EBITDA | | | | |
| EPS | | | | |

## Position (only if held)
Cost basis:
Current gain/loss %:
"""

STAGE_RULES = """## How to work this stage

1. Write ONLY the template sections listed above, in template order, under the
   template's own headings. Other sections are handled in other chats.
2. Use ONLY the material in this file. Do not use memory for company figures.
   If something the template asks for isn't here, write
   "Not in provided sources - open item" and move on.
3. Cite every hard number inline as [filename, accession]. For drafts, cite the
   original filing the draft cites, not the draft.
4. Do not do arithmetic. Use figures exactly as reported. Where the template
   wants a ratio, growth rate, or multiple that the sources don't state
   outright, do not compute it. Instead add a row to a table at the end titled
   "Calculations needed" with: metric | formula | each input with its value
   and citation. These get computed outside the chat and pasted back in.
5. Label allegations vs. confirmed findings every time.
6. End with a short "Open items" list: anything missing, ambiguous, or
   contradictory across sources.
"""

ASSEMBLY_RULES = """## How to work this stage

1. Write the listed new sections, using only the reviewed drafts below.
2. Then assemble the full post in template order: header block, sections 1-13.
   Take each drafted section as written; edit only for flow, repetition and
   consistency. Do not add new figures or claims during assembly.
3. Carry every citation through unchanged.
4. Merge every draft's "Calculations needed" and "Open items" lists into one
   appendix at the end, de-duplicated. Leave any uncomputed value visible as
   [CALC PENDING] in the body rather than filling it in.
5. List any place two drafts contradict each other, instead of picking one.
"""


# --------------------------------------------------------------------------
# Template parsing
# --------------------------------------------------------------------------

def parse_template(text: str) -> dict[str, str]:
    """Split the template into its level-2 sections, keyed by section number
    ("1".."13"), plus 'header', 'research' and 'style'."""
    out: dict[str, str] = {}
    chunks = re.split(r"(?m)^(?=## )", text)
    for ch in chunks:
        first = ch.split("\n", 1)[0]
        m = re.match(r"## (\d+)\.", first)
        body = ch.strip().rstrip("-").strip()
        if m:
            out[m.group(1)] = body
        elif first.lower().startswith("## header"):
            out["header"] = body
        elif first.lower().startswith("## research instructions"):
            out["research"] = body
    style = re.search(r"(?ms)^### Style notes.*?(?=^---|^## |\Z)", text)
    if style:
        out["style"] = style.group(0).strip()
        # Keep style notes out of section 13's body.
        if "13" in out:
            out["13"] = out["13"].split("### Style notes")[0].strip().rstrip("-").strip()
    return out


def template_sections(tpl: dict[str, str], wanted: list[str]) -> tuple[str, list[str]]:
    """Return the template text for the wanted sections. A subsection id like
    '9.2' pulls section 9 once, and the stage instructions name the
    subsections to write."""
    parts, missing, seen = [], [], set()
    for sid in wanted:
        top = sid.split(".")[0]
        if top in seen:
            continue
        if top not in tpl:
            missing.append(sid)
            continue
        seen.add(top)
        parts.append(tpl[top])
    return "\n\n---\n\n".join(parts), missing


# --------------------------------------------------------------------------
# Corpus index
# --------------------------------------------------------------------------

class Corpus:
    def __init__(self, root: Path):
        self.root = root
        mpath = root / "manifest.json"
        if not mpath.exists():
            sys.exit(f"No manifest.json in {root}. Point --corpus at the ticker "
                     "folder of a corpus fetched with --flat.")
        self.manifest = json.loads(mpath.read_text(encoding="utf-8"))
        self.ticker = self.manifest.get("ticker", root.name)
        self.rows = [r for r in self.manifest.get("filings", []) if r.get("file")]
        self.all_files = sorted(p.name for p in root.iterdir() if p.is_file())

    # Children are found by filename prefix rather than the manifest's
    # split_files/exhibit_files lists: a cached (non --force) run writes those
    # lists empty, and the files are still on disk.
    def children(self, row: dict) -> list[str]:
        stem = Path(row["file"]).stem + "_"
        return [f for f in self.all_files if f.startswith(stem)]

    def split_files(self, row: dict) -> dict[str, str]:
        out = {}
        for f in self.children(row):
            m = SPLIT_RE.search(f)
            if m and f[: m.start()] == Path(row["file"]).stem:
                out[m.group(1) + (m.group(2) or "")] = f
        return out

    def exhibit_files(self, row: dict) -> list[str]:
        return [f for f in self.children(row) if not SPLIT_RE.search(f)]

    def has_10q_since_latest_10k(self) -> bool:
        cutoff = _since_cutoff(self)
        return any((r.get("report_date") or "") > cutoff for r in self.of_form("10-Q"))

    def of_form(self, form: str) -> list[dict]:
        if form == "DEF 14A":
            rows = [r for r in self.rows if r["form"] in ("DEF 14A", "DEFR14A")]
        else:
            rows = [r for r in self.rows if r["form"].startswith(form)]
        key = "filing_date" if form in ("8-K", "DEF 14A") else "report_date"
        return sorted(rows, key=lambda r: (r.get(key) or r.get("filing_date") or ""),
                      reverse=True)


def _since_cutoff(corpus: "Corpus") -> str:
    tenk = corpus.of_form("10-K")
    return (tenk[0].get("report_date") or "") if tenk else ""


def items_of(row: dict) -> set[str]:
    return {i.strip() for i in (row.get("items") or "").split(",") if i.strip()}


def pick_rows(corpus: Corpus, sel: dict, warn: list[str]) -> list[dict]:
    form, which = sel["form"], sel.get("which", "latest")
    rows = corpus.of_form(form)

    if form == "8-K":
        if "items_any" in sel:
            want = set(sel["items_any"])
            rows = [r for r in rows if items_of(r) & want]
        if "items_other_than" in sel:
            common = set(sel["items_other_than"])
            rows = [r for r in rows if items_of(r) - common]

    if which == "latest":
        rows = rows[:1]
    elif which == "prior":
        rows = rows[1:2]
    elif which == "latest_n":
        rows = rows[: int(sel.get("n", 1))]
    elif which == "since_latest_10k":
        cutoff = _since_cutoff(corpus)
        rows = [r for r in rows if (r.get("report_date") or "") > cutoff]
        if not rows:
            warn.append(f"Info: no {form} after the latest 10-K period ({cutoff}), "
                        "so the 10-K is the most recent period.")
    elif which != "all":
        sys.exit(f"Unknown selector which={which!r}")

    if not rows and which != "since_latest_10k":
        warn.append(f"No {form} matched {sel}")
    return rows


def cover_text(md: str) -> str:
    m = ITEM_HEADING_RE.search(md)
    return (md[: m.start()] if m else md[:15000]).strip()


def row_label(row: dict) -> str:
    bits = [row["form"]]
    if row.get("report_date"):
        bits.append(f"period {row['report_date']}")
    bits.append(f"filed {row['filing_date']}")
    if row.get("items"):
        bits.append(f"items {row['items']}")
    bits.append(f"accession {row['accession']}")
    return " · ".join(bits)


def collect_sources(corpus: Corpus, sources: list[dict],
                    warn: list[str]) -> list[tuple[str, str, str]]:
    """Return (filename, provenance line, text), de-duplicated by content key."""
    out, seen = [], set()

    def add(key: str, name: str, prov: str, text: str):
        if key in seen:
            return
        seen.add(key)
        out.append((name, prov, text))

    for sel in sources:
        form = sel["form"]
        cond = sel.get("when")
        if cond:
            recent = corpus.has_10q_since_latest_10k()
            if (cond == "no_10q_since_latest_10k" and recent) or \
               (cond == "has_10q_since_latest_10k" and not recent):
                continue

        if form == "file":
            p = corpus.root / sel["name"]
            if p.exists():
                add(p.name, p.name, f"{p.name} · derived from Form 4/144 filings "
                    "(per-row accession numbers inside)", p.read_text(encoding="utf-8"))
            elif not sel.get("optional"):
                warn.append(f"Missing file: {sel['name']}")
            continue

        for row in pick_rows(corpus, sel, warn):
            prov = row_label(row)
            main = corpus.root / row["file"]
            part = sel.get("part")

            if form in ("10-K", "10-Q"):
                if part in ("cover", "full"):
                    if not main.exists():
                        warn.append(f"Missing {row['file']}")
                        continue
                    md = main.read_text(encoding="utf-8")
                    if part == "cover":
                        add(f"{row['file']}#cover", f"{main.stem} (cover page)",
                            prov, cover_text(md))
                    else:
                        add(row["file"], row["file"], prov, md)
                    continue

                splits = corpus.split_files(row)
                if not splits:
                    warn.append(f"{row['file']} has no split files; re-run the "
                                "fetch with --split.")
                    continue
                if form == "10-K":
                    dups = sorted(k for k in splits if "-" in k)
                    if dups and f"{row['file']}#dups" not in seen:
                        seen.add(f"{row['file']}#dups")
                        warn.append(f"{row['file']}: repeated Item heading(s) {dups}. A "
                                    "10-K has each Item once, so a cross-reference line "
                                    "was probably treated as a heading and cut a section "
                                    "short. Check before trusting the neighbouring Items.")
                for item in sel.get("items", []):
                    f = splits.get(item)
                    if not f:
                        warn.append(f"{row['file']}: Item {item} not found "
                                    f"(have: {', '.join(sorted(splits))})")
                        continue
                    text = (corpus.root / f).read_text(encoding="utf-8")
                    if form == "10-K" and item == "8" and len(text) < 40000:
                        warn.append(f"{f} is only {len(text):,} chars; the "
                                    "financial statements and notes may sit under "
                                    "Item 15. Check it and add \"15\" if so.")
                    add(f, f, prov, text)

            elif form == "8-K":
                if sel.get("main", True) and main.exists():
                    add(row["file"], row["file"], prov,
                        main.read_text(encoding="utf-8"))
                skip = set(sel.get("exhibits_skip_if_items", []))
                if sel.get("exhibits", True) and not (items_of(row) & skip):
                    exs = corpus.exhibit_files(row)
                    if not exs and "2.02" in items_of(row) and sel.get("exhibits"):
                        warn.append(f"{row['file']}: Item 2.02 with no exhibit file")
                    for f in exs:
                        add(f, f, prov + " · exhibit",
                            (corpus.root / f).read_text(encoding="utf-8"))

            else:  # DEF 14A
                if main.exists():
                    add(row["file"], row["file"], prov,
                        main.read_text(encoding="utf-8"))
                else:
                    warn.append(f"Missing {row['file']}")
    return out


# --------------------------------------------------------------------------
# Bundle assembly
# --------------------------------------------------------------------------

def build_bundle(b: dict, corpus: Corpus, tpl: dict[str, str],
                 drafts_dir: Path, inputs_dir: Path) -> tuple[str, dict]:
    warn: list[str] = []
    tpl_text, missing = template_sections(tpl, b.get("sections", []))
    if missing:
        warn.append(f"Template sections not found: {missing}")

    sources = collect_sources(corpus, b.get("sources", []), warn)

    drafts = []
    for d in b.get("drafts", []):
        p = drafts_dir / f"{d}.md"
        if p.exists():
            drafts.append((d, p.read_text(encoding="utf-8")))
        else:
            warn.append(f"Draft {d} missing ({p}). Run bundle {d} first and "
                        "save the reviewed reply there.")

    inputs = []
    for name in b.get("inputs", []):
        p = inputs_dir / name
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            inputs.append((name, txt))
            if name == "market_data.md" and "| Price | | |" in txt:
                warn.append("market_data.md looks unfilled (Price is blank).")
        elif name == "market_data.md":
            warn.append(f"market_data.md not created yet ({p}); a run without "
                        "--check creates the blank form.")
        else:
            warn.append(f"Input {name} missing ({p}).")

    sec_list = ", ".join(b.get("sections", [])) or "none (task stage)"
    today = date.today().isoformat()
    L = [f"# {corpus.ticker} - Stage {b['id']}: {b['title']}", "",
         f"*Bundle built {today} by bundle.py from manifest generated "
         f"{corpus.manifest.get('generated_at', '?')[:10]}.*", "",
         (f"**Task:** {b['task']}" if b.get("task")
          else f"**Write these template sections: {sec_list}**"), ""]
    if b.get("notes"):
        L += [f"> Note: {b['notes']}", ""]
    rules = ASSEMBLY_RULES if b.get("assemble") else STAGE_RULES
    if b.get("task"):
        rules = rules.replace("Write ONLY the template sections listed above, in template order, under the\n   template's own headings. Other sections are handled in other chats.",
                              "Do only the task stated above. Template sections are written in other chats.")
    L += [rules, ""]
    if warn:
        L += ["## Gaps known before this chat started", ""]
        L += [f"- {w}" for w in warn] + [""]

    L += ["## Contents", "", "| # | Source | Provenance |", "|---|---|---|"]
    n = 0
    for name, _ in inputs:
        n += 1
        L.append(f"| {n} | {name} | hand-entered input |")
    for d, _ in drafts:
        n += 1
        L.append(f"| {n} | drafts/{d}.md | reviewed draft |")
    for name, prov, _ in sources:
        n += 1
        L.append(f"| {n} | `{name}` | {prov} |")
    L += [""]

    if tpl_text:
        L += ["=" * 70, "# PART 1 - TEMPLATE SECTIONS TO WRITE", "=" * 70, "",
              tpl_text, ""]
    if tpl.get("research"):
        L += ["---", "", tpl["research"], ""]
    if tpl.get("style"):
        L += ["---", "", tpl["style"], ""]

    if inputs:
        L += ["=" * 70, "# PART 2 - HAND-ENTERED INPUTS", "=" * 70, ""]
        for name, txt in inputs:
            L += [f"<<<INPUT {name}>>>", txt.strip(), f"<<<END INPUT {name}>>>", ""]
    if drafts:
        L += ["=" * 70, "# PART 3 - REVIEWED DRAFTS", "=" * 70, ""]
        for d, txt in drafts:
            L += [f"<<<DRAFT {d}>>>", txt.strip(), f"<<<END DRAFT {d}>>>", ""]
    if sources:
        L += ["=" * 70, "# PART 4 - SOURCE FILINGS", "=" * 70, ""]
        for name, prov, txt in sources:
            L += [f"<<<SOURCE {name}>>>", f"*{prov}*", "", txt.strip(),
                  f"<<<END SOURCE {name}>>>", ""]

    text = "\n".join(L) + "\n"
    stats = {
        "id": b["id"], "title": b["title"], "sections": sec_list,
        "sources": len(sources), "drafts": len(drafts),
        "tokens": int(len(text) / CHARS_PER_TOKEN), "warnings": warn,
        "per_source": [(name, int(len(t) / CHARS_PER_TOKEN)) for name, _, t in sources],
    }
    return text, stats


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path,
                    help="Ticker folder of a --flat corpus (contains manifest.json)")
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=here / "bundles.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output folder (default: <corpus>/bundles)")
    ap.add_argument("--drafts", type=Path, default=None,
                    help="Folder of reviewed drafts A.md..F.md (default: <out>/drafts)")
    ap.add_argument("--only", default="",
                    help="Comma-separated bundle ids to build, e.g. E or A,B")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--check", action="store_true",
                    help="Report sizes and gaps without writing bundles")
    ap.add_argument("--detail", action="store_true",
                    help="List every source with its token estimate, not just the top 5 when over")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    limit = args.max_tokens or cfg.get("max_tokens", 150000)
    corpus = Corpus(args.corpus)
    tpl = parse_template(args.template.read_text(encoding="utf-8"))
    out = args.out or (args.corpus / "bundles")
    drafts_dir = args.drafts or (out / "drafts")

    if not args.check:
        drafts_dir.mkdir(parents=True, exist_ok=True)
        md = out / "market_data.md"
        if not md.exists():
            md.write_text(MARKET_DATA_TEMPLATE.format(ticker=corpus.ticker),
                          encoding="utf-8")
            print(f"Created {md} - fill it in before building bundle E.")

    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    report = [f"# {corpus.ticker} bundle report - {date.today().isoformat()}", "",
              f"Token estimate = chars / {CHARS_PER_TOKEN}; limit {limit:,}.", "",
              "| Bundle | Sections | Sources | Drafts | ~Tokens | Status |",
              "|---|---|---|---|---|---|"]
    details = []

    for b in cfg["bundles"]:
        if only and b["id"].upper() not in only:
            continue
        text, st = build_bundle(b, corpus, tpl, drafts_dir, out)
        over = st["tokens"] > limit
        real = [w for w in st["warnings"] if not w.startswith("Info:")]
        status = "OVER LIMIT" if over else ("gaps" if real else "ok")
        report.append(f"| {st['id']} {st['title']} | {st['sections']} | "
                      f"{st['sources']} | {st['drafts']} | {st['tokens']:,} | {status} |")

        print(f"\n[{st['id']}] {st['title']}  ~{st['tokens']:,} tokens  {status}")
        for w in st["warnings"]:
            print(f"    ! {w}")
        if over or args.detail:
            ranked = sorted(st["per_source"], key=lambda x: -x[1])
            shown = ranked if args.detail else ranked[:5]
            print(f"    Sources ({len(ranked)}, largest first):")
            for name, t in shown:
                print(f"      {t:>8,}  {name}")
            if len(shown) < len(ranked):
                rest = sum(t for _, t in ranked[len(shown):])
                print(f"      {rest:>8,}  ...{len(ranked) - len(shown)} more (use --detail)")

        details += [f"## {st['id']} - {st['title']}", ""]
        details += [f"- ~{t:,} tokens: `{name}`" for name, t in st["per_source"]]
        details += [f"- WARNING: {w}" for w in st["warnings"]] + [""]

        if not args.check:
            out.mkdir(parents=True, exist_ok=True)
            fname = f"{corpus.ticker}_stage{st['id']}_{re.sub(r'[^A-Za-z0-9]+', '-', st['title']).strip('-').lower()}.md"
            (out / fname).write_text(text, encoding="utf-8")

    if not args.check:
        (out / "bundle_report.md").write_text("\n".join(report + [""] + details) + "\n",
                                              encoding="utf-8")
        print(f"\nWrote bundles and bundle_report.md to {out}")


if __name__ == "__main__":
    main()
