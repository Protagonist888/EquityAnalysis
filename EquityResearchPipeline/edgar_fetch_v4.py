#!/usr/bin/env python3
r"""
edgar_fetch.py — Build a frozen, citable SEC filing corpus for equity research.


To run: Create vitual environment, install dependencies, then run the script with your email and a list of tickers.
Command prompts to run: 
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install requests beautifulsoup4 lxml
python edgar_fetch.py --tickers GTLB --email markchung21@hotmail.com --years 2 --flat --split --force --out C:\Users\markc\Downloads\corpus-flat



Downloads and converts, for each ticker, over a trailing N-year window:
  10-K, 10-Q      -> markdown (text + tables)
  8-K             -> markdown, FILTERED to material items only, incl. EX-99.x exhibits
  DEF 14A / DEFA  -> markdown
  Form 4 / 5      -> one CSV per company (parsed XML, not prose)
  Form 144        -> one CSV per company

Every filing gets a meta.json carrying the accession number, so any figure in a
downstream analysis can be traced to its source document.

Usage:
    python edgar_fetch.py --tickers GTLB,TEAM,FROG --email you@example.com
    python edgar_fetch.py --tickers GTLB --years 2 --out ./corpus
    python edgar_fetch.py --tickers GTLB --forms 10-K,10-Q,8-K
    python edgar_fetch.py --tickers GTLB --dry-run        # list, don't download

Requires: requests, beautifulsoup4, lxml
    pip install requests beautifulsoup4 lxml
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import warnings

import requests
from bs4 import BeautifulSoup

try:  # Inline XBRL filings are XHTML; the HTML parser handles them fine.
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEC_RATE_LIMIT = 8.0  # requests/sec; SEC's published ceiling is 10
TIMEOUT = 30
MAX_RETRIES = 4

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"

# 8-K items worth collecting. Everything else is noise for this workflow.
# See guideline doc §3 for the reasoning on each.
DEFAULT_8K_ITEMS = {
    "1.01",  # Entry into a Material Definitive Agreement
    "1.02",  # Termination of a Material Definitive Agreement
    "2.02",  # Results of Operations — the big one; EX-99.1 carries guidance
    "3.02",  # Unregistered Sales of Equity Securities (dilution)
    "4.01",  # Change in Certifying Accountant (auditor change)
    "4.02",  # Non-Reliance on Previously Issued Financials (restatement)
    "5.02",  # Departure/Election of Directors or Officers
    # Landmine items: rare, and each one is a warning on its own.
    "1.03",  # Bankruptcy or receivership
    "1.05",  # Material cybersecurity incident
    "2.03",  # Creation of a direct financial obligation (new debt)
    "2.04",  # Triggering events that accelerate an obligation (default)
    "2.05",  # Costs of exit or disposal activities (restructuring)
    "2.06",  # Material impairments
    "3.01",  # Delisting notice / failure to meet listing standard
}

FORM_GROUPS = {
    "10-K": ["10-K", "10-K/A", "10-KT"],
    "10-Q": ["10-Q", "10-Q/A"],
    "8-K": ["8-K", "8-K/A"],
    "DEF 14A": ["DEF 14A", "DEFA14A", "DEFR14A"],
    "4": ["4", "4/A", "5", "5/A"],
    "144": ["144", "144/A"],
}

# Form 4 transaction codes. Only P and S carry real signal; the rest are
# compensation mechanics. See guideline doc §4.
SIGNAL_CODES = {"P", "S"}
CODE_MEANINGS = {
    "P": "Open-market purchase",
    "S": "Open-market sale",
    "A": "Grant/award",
    "M": "Option exercise",
    "F": "Shares withheld for taxes",
    "G": "Gift",
    "C": "Conversion",
    "D": "Disposition to issuer",
    "X": "Option exercise (in/out of money)",
    "J": "Other (see footnotes)",
}


# --------------------------------------------------------------------------
# HTTP with rate limiting
# --------------------------------------------------------------------------

class SecClient:
    """Rate-limited SEC client. The User-Agent is mandatory — without a real
    contact address the SEC blocks you, and rightly so."""

    def __init__(self, email: str, app_name: str = "equity-research-corpus"):
        if not email or "@" not in email:
            raise ValueError("A real contact email is required by the SEC.")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"{app_name} ({email})",
            "Accept-Encoding": "gzip, deflate",
        })
        self._min_interval = 1.0 / SEC_RATE_LIMIT
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def get(self, url: str, as_json: bool = False):
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                r = self.session.get(url, timeout=TIMEOUT)
            except requests.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
                continue

            if r.status_code == 200:
                return r.json() if as_json else r.text
            if r.status_code == 404:
                return None
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt + 1)
                continue
            r.raise_for_status()
        return None


# --------------------------------------------------------------------------
# HTML -> Markdown (tables preserved)
# --------------------------------------------------------------------------

# Cells that are pure decoration in SEC financial tables.
_DECORATION = {"", "$", "%", "(", ")", "—", "–", "-", "|"}


def _clean_cells(cells: list[str]) -> list[str]:
    """SEC HTML splits '$ (1,234 )' across three or four <td>s. Merge the
    fragments back into single values so the row reads like the printed page.

    Critically, this returns a list of the SAME LENGTH as the input. Decoration
    cells are blanked in place rather than removed, because removing them
    shifts every column to its right — and a financial table whose columns are
    off by one is worse than no table at all, since it looks correct. Empty
    columns are dropped later, table-wide, where alignment is preserved."""
    n = len(cells)
    out = [""] * n
    pending_prefix = ""   # '$' and/or '(' waiting to attach to the next value
    last_value_idx = -1

    for i, raw in enumerate(cells):
        c = re.sub(r"\s+", " ", raw.replace("\u00a0", " ")).strip()

        if c == "$":
            pending_prefix += "$"
            continue
        if c == "(":
            pending_prefix += "("
            continue
        if c == ")":
            # Close a negative whose parenthesis was split off from its number.
            # lstrip('$') because the value may already carry a currency prefix.
            if last_value_idx >= 0:
                v = out[last_value_idx]
                if v.lstrip("$").startswith("(") and not v.endswith(")"):
                    out[last_value_idx] = v + ")"
            continue
        if c in {"", "|", "—", "–"}:
            if c in {"—", "–"}:
                out[i] = "—"
                last_value_idx = i
            continue

        # A value: re-attach whatever decoration was waiting.
        if "(" in pending_prefix:
            val = "(" + c
            if "$" in pending_prefix:
                val = "$" + val
        else:
            val = ("$" + c) if "$" in pending_prefix else c
        out[i] = val
        last_value_idx = i
        pending_prefix = ""

    return out


def _compact_rows(rows: list[list[str]]) -> list[list[str]]:
    """Collapse each row to [label] + [values in order], then pad to a common
    width.

    SEC HTML gives different rows different physical cell counts for the same
    logical columns: '$ 286,254' is two cells, '( 56,935 )' is three. Aligning
    on physical position therefore misaligns rows against each other. Aligning
    on the *sequence* of non-empty values is far more robust for financial
    statements, where column N is simply the Nth period."""
    compacted = []
    for r in rows:
        label = r[0].strip() if r else ""
        values = [c for c in r[1:] if c.strip()]
        compacted.append([label] + values)

    width = max((len(r) for r in compacted), default=0)
    if width < 2:
        return []
    return [r + [""] * (width - len(r)) for r in compacted]


def _table_to_md(table) -> str:
    """Convert one <table> to markdown. Returns '' for layout tables."""
    rows = []
    for tr in table.find_all("tr", recursive=True):
        cells = []
        for td in tr.find_all(["td", "th"], recursive=False):
            text = td.get_text(" ", strip=True)
            # Expand colspan so every row has the same physical column count.
            try:
                span = max(1, int(td.get("colspan", 1)))
            except (TypeError, ValueError):
                span = 1
            cells.append(text)
            cells.extend([""] * (span - 1))
        if not cells:
            continue
        cleaned = _clean_cells(cells)
        if any(c.strip() for c in cleaned):
            rows.append(cleaned)

    if len(rows) < 2:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    rows = _compact_rows(rows)
    if not rows or len(rows) < 2:
        return ""
    width = len(rows[0])
    if width < 2:
        return ""

    # Layout-table heuristic: a real financial table has numbers in it.
    numericish = sum(
        1 for r in rows for c in r
        if re.search(r"\d", c) and not re.fullmatch(r"\(?\d{4}\)?", c.strip())
    )
    if numericish < 2:
        return ""

    header, body = rows[0], rows[1:]

    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


ITEM_RE = re.compile(r"^\s*ITEM\s+\d+[A-Z]?\.?\s*[-–—:]?\s*\S", re.IGNORECASE)


def html_to_markdown(html: str) -> tuple[str, dict]:
    """Convert a SEC HTML filing to markdown. Returns (markdown, report)."""
    soup = BeautifulSoup(html, "lxml")

    # Inline-XBRL hidden header holds hundreds of invisible facts. Strip it.
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:header$", re.I)):
        tag.decompose()
    for tag in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.I)}):
        tag.decompose()

    # Process innermost tables first so layout wrappers don't swallow content.
    all_tables = soup.find_all("table")
    inner_tables = [t for t in all_tables if not t.find("table")]
    tables_kept = 0

    for t in inner_tables:
        md = _table_to_md(t)
        if md:
            tables_kept += 1
            placeholder = soup.new_string(f"\n\n{md}\n\n")
            t.replace_with(placeholder)
        else:
            # Layout table: keep its text, lose its structure.
            t.replace_with(soup.new_string(" " + t.get_text(" ", strip=True) + " "))

    # Remaining (outer) tables: unwrap to text.
    for t in soup.find_all("table"):
        t.replace_with(soup.new_string(" " + t.get_text(" ", strip=True) + " "))

    text = soup.get_text("\n")
    text = text.replace("\u00a0", " ")

    lines = []
    for ln in text.split("\n"):
        s = re.sub(r"[ \t]+", " ", ln).strip()
        if not s:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if ITEM_RE.match(s) and len(s) < 200:
            s = "## " + s
        lines.append(s)

    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"

    report = {
        "tables_found": len(all_tables),
        "tables_kept": tables_kept,
        "chars": len(md),
        "item_headings": sum(1 for l in lines if l.startswith("## ")),
    }
    return md, report


# --------------------------------------------------------------------------
# Form 4 / 5 XML parsing
# --------------------------------------------------------------------------

def _strip_ns(xml: str) -> str:
    return re.sub(r'\sxmlns(:\w+)?="[^"]*"', "", xml, count=0)


def _txt(node, path, default=""):
    if node is None:
        return default
    el = node.find(path)
    return el.get_text(strip=True) if el else default


def parse_form4(xml: str, accession: str, filing_date: str) -> list[dict]:
    """Parse an ownership XML document into one row per transaction."""
    soup = BeautifulSoup(_strip_ns(xml), "xml")
    doc = soup.find("ownershipDocument")
    if doc is None:
        return []

    owner = doc.find("reportingOwner")
    name = _txt(owner, "rptOwnerName")
    rel = owner.find("reportingOwnerRelationship") if owner else None
    is_dir = _txt(rel, "isDirector") in {"1", "true"}
    is_off = _txt(rel, "isOfficer") in {"1", "true"}
    is_ten = _txt(rel, "isTenPercentOwner") in {"1", "true"}
    title = _txt(rel, "officerTitle")

    role = title or ("Director" if is_dir else "10% owner" if is_ten else "Insider")

    # Rule 10b5-1: a scheduled sale carries far less signal than a
    # discretionary one. Check the structured flag, then fall back to
    # footnote text, because older filings only say it in prose.
    aff = doc.find("aff10b5One")
    plan_flag = bool(aff and aff.get_text(strip=True) in {"1", "true"})
    # Footnotes carry the detail that makes a transaction interpretable: which
    # fund vehicle held the shares, whether a purchase was under an ESPP,
    # whether a sale was a sell-to-cover. Keep the text, don't just scan it.
    footnote_text = " ".join(f.get_text(" ", strip=True)
                             for f in doc.find_all("footnote"))
    remarks = _txt(doc, "remarks")
    if remarks:
        footnote_text = (footnote_text + " " + remarks).strip()
    if re.search(r"10b5[\s\-–]?1", footnote_text, re.I):
        plan_flag = True

    rows = []
    for table_name, is_deriv in (("nonDerivativeTable", False),
                                 ("derivativeTable", True)):
        tbl = doc.find(table_name)
        if not tbl:
            continue
        tag = "nonDerivativeTransaction" if not is_deriv else "derivativeTransaction"
        for tx in tbl.find_all(tag):
            coding = tx.find("transactionCoding")
            code = _txt(coding, "transactionCode")
            amounts = tx.find("transactionAmounts")
            shares = _txt(amounts, "transactionShares")
            price = _txt(amounts, "transactionPricePerShare")
            ad = _txt(amounts, "transactionAcquiredDisposedCode")
            post = _txt(tx.find("postTransactionAmounts"),
                        "sharesOwnedFollowingTransaction")

            def num(s):
                try:
                    return float(str(s).replace(",", "").replace("$", ""))
                except (ValueError, TypeError):
                    return None

            sh, pr = num(shares), num(price)
            rows.append({
                "filing_date": filing_date,
                "accession_number": accession,
                "insider_name": name,
                "insider_role": role,
                "is_director": is_dir,
                "is_officer": is_off,
                "is_ten_pct_owner": is_ten,
                "transaction_date": _txt(tx.find("transactionDate"), "value"),
                "security_title": _txt(tx.find("securityTitle"), "value"),
                "is_derivative": is_deriv,
                "transaction_code": code,
                "code_meaning": CODE_MEANINGS.get(code, "Unknown"),
                "acquired_disposed": ad,
                "shares": sh,
                "price_per_share": pr,
                "total_value": round(sh * pr, 2) if (sh is not None and pr) else None,
                "shares_owned_after": num(post),
                # Fraction of the holding this sale represents. 1.0 means the
                # position went to zero — the tell for a fund vehicle
                # completing an exit rather than an insider trimming.
                "pct_of_position_sold": (
                    round(sh / (sh + num(post)), 4)
                    if (code == "S" and sh and num(post) is not None
                        and (sh + num(post)) > 0) else None
                ),
                "is_10b5_1": plan_flag,
                "is_signal_code": code in SIGNAL_CODES,
                "footnotes": footnote_text[:1500],
            })
    return rows


def parse_form144(xml: str, accession: str, filing_date: str) -> list[dict]:
    """Parse an electronic Form 144 (mandatory on EDGAR since April 2023).
    Schema varies more than Form 4, so this is deliberately forgiving."""
    soup = BeautifulSoup(_strip_ns(xml), "xml")
    rows = []

    issuer = _txt(soup, "issuerName") or _txt(soup, "nameOfIssuer")
    person = (_txt(soup, "personSellingName")
              or _txt(soup, "nameOfPersonForWhoseAccount")
              or _txt(soup, "filerFullName"))

    blocks = soup.find_all("securitiesToBeSold") or [soup]
    for b in blocks:
        def num(s):
            try:
                return float(str(s).replace(",", "").replace("$", ""))
            except (ValueError, TypeError):
                return None

        rows.append({
            "filing_date": filing_date,
            "accession_number": accession,
            "issuer": issuer,
            "person_selling": person,
            "security_class": _txt(b, "securityClassTitle") or _txt(b, "classOfSecurities"),
            "shares_to_be_sold": num(_txt(b, "numberOfSharesToBeSold")
                                     or _txt(b, "unitsToBeSold")),
            "aggregate_market_value": num(_txt(b, "aggregateMarketValue")),
            "approx_sale_date": _txt(b, "approxSaleDate") or _txt(soup, "approxSaleDate"),
            "exchange": _txt(b, "nameOfExchange") or _txt(soup, "nameOfExchange"),
        })
    return rows


def summarize_form4(signal_rows: list[dict]) -> str:
    """Condense open-market activity into the handful of facts §9.1 needs.

    Sales under a pre-arranged Rule 10b5-1 plan are scheduled rather than
    discretionary and carry much less signal, so they are counted separately.
    Conflating the two is the most common way insider analysis goes wrong."""
    if not signal_rows:
        return "# Insider activity (open-market)\n\nNo open-market purchases or sales in the window.\n"

    by_q: dict[str, dict] = {}
    for r in signal_rows:
        d = r.get("transaction_date") or r.get("filing_date") or ""
        if len(d) < 7:
            continue
        y, m = int(d[:4]), int(d[5:7])
        q = f"{y}Q{(m - 1) // 3 + 1}"
        b = by_q.setdefault(q, {"buy_sh": 0.0, "buy_val": 0.0,
                                "sell_sh": 0.0, "sell_val": 0.0,
                                "plan_sell_val": 0.0,
                                "buyers": set(), "sellers": set()})
        sh = r.get("shares") or 0.0
        val = r.get("total_value") or 0.0
        if r["transaction_code"] == "P":
            b["buy_sh"] += sh
            b["buy_val"] += val
            b["buyers"].add(r["insider_name"])
        else:
            b["sell_sh"] += sh
            b["sell_val"] += val
            b["sellers"].add(r["insider_name"])
            if r.get("is_10b5_1"):
                b["plan_sell_val"] += val

    lines = ["# Insider activity (open-market only, codes P and S)", "",
             "| Quarter | Buy $ | Buyers | Sell $ | Sellers | Of which 10b5-1 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for q in sorted(by_q, reverse=True):
        b = by_q[q]
        plan_pct = (b["plan_sell_val"] / b["sell_val"] * 100) if b["sell_val"] else 0
        lines.append(
            f"| {q} | ${b['buy_val']:,.0f} | {len(b['buyers'])} | "
            f"${b['sell_val']:,.0f} | {len(b['sellers'])} | {plan_pct:.0f}% |"
        )

    discretionary = [r for r in signal_rows
                     if r["transaction_code"] == "P" or not r.get("is_10b5_1")]
    lines += ["", "## Worth a look", ""]
    buys = [r for r in signal_rows if r["transaction_code"] == "P"]
    if buys:
        lines.append(f"- {len(buys)} open-market purchase(s) by "
                     f"{len({r['insider_name'] for r in buys})} insider(s). "
                     "Purchases are the high-signal event here.")
    else:
        lines.append("- No open-market purchases in the window.")
    disc_sales = [r for r in signal_rows
                  if r["transaction_code"] == "S" and not r.get("is_10b5_1")]
    if disc_sales:
        lines.append(f"- {len(disc_sales)} discretionary sale(s) (not under a "
                     "10b5-1 plan) — these carry more signal than scheduled sales.")

    # Large position reductions: a sale leaving the insider much smaller.
    for r in signal_rows:
        after, sh = r.get("shares_owned_after"), r.get("shares")
        if r["transaction_code"] == "S" and after and sh and (sh / (after + sh)) > 0.5:
            lines.append(f"- {r['insider_name']} ({r['insider_role']}) sold "
                         f"{sh:,.0f} shares on {r['transaction_date']}, more than "
                         f"half their position.")
    lines.append("")
    lines.append("*Codes A (grants), M (option exercises) and F (tax withholding) "
                 "are excluded — they are compensation mechanics, not decisions. "
                 "See form4_transactions.csv for the full record.*")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Filing discovery
# --------------------------------------------------------------------------

@dataclass
class Filing:
    cik: str
    ticker: str
    form: str
    accession: str
    filing_date: str
    report_date: str
    primary_doc: str
    items: str = ""

    @property
    def acc_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def base_url(self) -> str:
        return ARCHIVE_BASE.format(cik=int(self.cik), acc_nodash=self.acc_nodash)

    @property
    def primary_url(self) -> str:
        return f"{self.base_url}/{self.primary_doc}"


def resolve_ciks(client: SecClient, tickers: list[str]) -> dict[str, str]:
    data = client.get(TICKER_MAP_URL, as_json=True)
    lookup = {}
    for row in data.values():
        lookup[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    out = {}
    for t in tickers:
        cik = lookup.get(t.upper())
        if cik:
            out[t.upper()] = cik
        else:
            print(f"  ! Ticker not found in SEC map: {t}", file=sys.stderr)
    return out


def list_filings(client: SecClient, ticker: str, cik: str,
                 since: date, wanted_forms: set[str]) -> list[Filing]:
    """Walk the submissions JSON, including older pages. The 'recent' block
    holds ~1000 filings — enough for most companies over two years, but a
    heavy Form 4 filer can overflow it, hence the pagination."""
    subs = client.get(SUBMISSIONS_URL.format(cik10=cik), as_json=True)
    if not subs:
        return []

    blocks = [subs["filings"]["recent"]]
    for extra in subs["filings"].get("files", []):
        # Older pages are only worth fetching if they overlap the window.
        if extra.get("filingTo", "") >= since.isoformat():
            page = client.get(f"https://data.sec.gov/submissions/{extra['name']}",
                              as_json=True)
            if page:
                blocks.append(page)

    results = []
    for block in blocks:
        n = len(block.get("accessionNumber", []))
        for i in range(n):
            fdate = block["filingDate"][i]
            if fdate < since.isoformat():
                continue
            form = block["form"][i]
            if form not in wanted_forms:
                continue
            results.append(Filing(
                cik=cik,
                ticker=ticker,
                form=form,
                accession=block["accessionNumber"][i],
                filing_date=fdate,
                report_date=block.get("reportDate", [""] * n)[i] or "",
                primary_doc=block.get("primaryDocument", [""] * n)[i] or "",
                items=block.get("items", [""] * n)[i] or "",
            ))
    results.sort(key=lambda f: f.filing_date, reverse=True)
    return results


def keep_8k(filing: Filing, wanted_items: set[str]) -> bool:
    """An 8-K without item filtering is mostly noise. Item 9.01 (Exhibits) is
    attached to nearly everything and is never a reason to keep a filing."""
    if not filing.items:
        return False
    present = {i.strip() for i in filing.items.split(",") if i.strip()}
    return bool(present & wanted_items)


# Filers name EX-99 exhibits every which way: "ex991.htm", "ex-99_1.htm",
# "gtlb-20260602xex991.htm", "d123456dex991.htm", "a991pressrelease.htm".
# Anchoring on the start of the filename misses most of them, which silently
# drops the earnings release — the whole reason for collecting Item 2.02.
EX99_NAME_RE = re.compile(r"(ex|exhibit)[-_ ]?99", re.I)
DOC_EXT = (".htm", ".html", ".txt")


def find_exhibits(client: SecClient, filing: Filing) -> list[tuple[str, str]]:
    """Locate EX-99.x exhibits — for an Item 2.02 8-K this is the earnings
    press release, where forward guidance actually lives.

    Two passes: a permissive filename match, then the filing index page, which
    carries an authoritative Type column for filers whose naming defeats the
    regex entirely."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    idx = client.get(f"{filing.base_url}/index.json", as_json=True)
    if idx:
        for item in idx.get("directory", {}).get("item", []):
            nm = item.get("name", "")
            if nm.lower().endswith(DOC_EXT) and EX99_NAME_RE.search(nm):
                if nm not in seen:
                    seen.add(nm)
                    out.append((nm, f"{filing.base_url}/{nm}"))

    if out:
        return out

    # Fallback: the human-facing index page lists each document with its Type.
    page = client.get(f"{filing.base_url}/{filing.accession}-index.htm")
    if not page:
        return out
    soup = BeautifulSoup(page, "lxml")
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
        if not any(re.match(r"^EX-99", c, re.I) for c in cells):
            continue
        link = row.find("a", href=True)
        if not link:
            continue
        href = link["href"]
        nm = href.rsplit("/", 1)[-1]
        # iXBRL viewer links need unwrapping to reach the raw document.
        nm = re.sub(r"^.*?doc=.*?/", "", nm)
        if nm.lower().endswith(DOC_EXT) and nm not in seen:
            seen.add(nm)
            url = href if href.startswith("http") else f"https://www.sec.gov{href}" \
                if href.startswith("/") else f"{filing.base_url}/{nm}"
            out.append((nm, url))
    return out


def find_ownership_xml(client: SecClient, filing: Filing) -> str | None:
    """The submissions feed points at the XSL-rendered view; the raw XML sits
    alongside it. Strip the xslF345X0N/ prefix, then fall back to a listing."""
    pdoc = filing.primary_doc
    if pdoc:
        candidate = re.sub(r"^xsl[^/]*/", "", pdoc)
        if candidate.lower().endswith(".xml"):
            return f"{filing.base_url}/{candidate}"
    idx = client.get(f"{filing.base_url}/index.json", as_json=True)
    if not idx:
        return None
    for item in idx.get("directory", {}).get("item", []):
        nm = item.get("name", "")
        if nm.lower().endswith(".xml") and "filingsummary" not in nm.lower():
            return f"{filing.base_url}/{nm}"
    return None


# --------------------------------------------------------------------------
# Download orchestration
# --------------------------------------------------------------------------

def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


ITEM_SPLIT_RE = re.compile(
    r"^##\s*ITEM\s+(\d+[A-Z]?)\.?\s*[-–—:]?\s*(.*)$", re.IGNORECASE)

# Minimum section size, used only when a filing's headings can't be matched
# to the standard Item sequence. Table-of-contents lines match the heading
# pattern too but produce fragments of a few dozen characters.
MIN_SECTION_CHARS = 1500

# The order Items appear in. A heading that doesn't fit this order is a
# cross-reference that the converter put on its own line (hyperlinked
# references like: see "Item 7. Management's Discussion ..."), not a section.
CANON_10K = ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
             "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16"]
# Part I then Part II; Item numbers restart in Part II.
CANON_10Q = ["1", "2", "3", "4", "1", "1A", "2", "3", "4", "5", "6"]

TOC_SPAN_CHARS = 400


def _align_to_canon(labels: list[str], canon: list[str]) -> list[int]:
    """Longest common subsequence of heading labels and the standard Item
    order. Returns indices of the headings to keep. On ties the earliest
    heading wins, so a later self-reference can't displace the real one."""
    n, m = len(labels), len(canon)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if labels[i] == canon[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    keep, i, j = [], 0, 0
    while i < n and j < m:
        if labels[i] == canon[j] and dp[i][j] == 1 + dp[i + 1][j + 1]:
            keep.append(i)
            i, j = i + 1, j + 1
        elif dp[i][j + 1] >= dp[i + 1][j]:
            j += 1
        else:
            i += 1
    return keep


def split_by_item(md: str, form: str = "10-K") -> list[tuple[str, str, str]]:
    """Split a 10-K or 10-Q into its Item sections.

    Returns (item_label, title_slug, text). A 465KB 10-K is most of a context
    window on its own; Item 1 + 1A + 7 + 8 is the part the analysis template
    actually reads, and that's a third of the size.

    Headings are kept only where they fit the standard Item order. Without
    that check, a cross-reference line inside Item 1 reading 'Item 7. MD&A'
    cuts Item 1 short and files the rest of it as Item 7."""
    lines = md.split("\n")
    marks: list[tuple[int, str, str]] = []
    for i, ln in enumerate(lines):
        m = ITEM_SPLIT_RE.match(ln)
        if not m:
            continue
        # Real headings never contain double quotes; quoted cross-references do.
        if any(q in ln for q in ('"', "\u201c", "\u201d")):
            continue
        num = m.group(1).upper()
        title = slug(m.group(2).strip().lower())[:48].strip("-")
        marks.append((i, num, title))
    if not marks:
        return []

    def span(k: int) -> int:
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        return len("\n".join(lines[marks[k][0]:end]))

    # Drop a leading table of contents: a run of three or more headings with
    # almost no text between them.
    lead = 0
    while lead < len(marks) and span(lead) < TOC_SPAN_CHARS:
        lead += 1
    if lead >= 3:
        marks = marks[lead:]
    if not marks:
        return []

    canon = CANON_10Q if form.startswith("10-Q") else CANON_10K
    keep = _align_to_canon([mk[1] for mk in marks], canon)
    aligned = len(keep) >= 3
    if aligned:
        marks = [marks[k] for k in keep]

    sections = []
    for j, (start, num, title) in enumerate(marks):
        end = marks[j + 1][0] if j + 1 < len(marks) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        # Aligned headings are real, so short sections ("Item 9. None.") are
        # kept: a stated "None" is evidence. Unaligned output keeps the old
        # size floor to shed table-of-contents fragments.
        if not aligned and len(body) < MIN_SECTION_CHARS:
            continue
        if aligned and body.count("\n") == 0:
            continue
        sections.append((num, title, body))

    # A 10-Q carries Item 1 in both Part I (financials) and Part II (legal).
    # Label repeats by occurrence: 1 = Part I, 1-2 = Part II.
    counts: dict[str, int] = {}
    out = []
    for num, title, body in sections:
        counts[num] = counts.get(num, 0) + 1
        label = num if counts[num] == 1 else f"{num}-{counts[num]}"
        out.append((label, title, body))
    return out


def write_splits(d: Path, base: str, md: str, form: str) -> list[str]:
    """Write per-Item files for one filing, replacing any from an earlier run
    (a stale 'item7-2' file would otherwise survive a corrected split)."""
    for old in d.glob(f"{base}_item*.md"):
        old.unlink()
    names = []
    for label, title, body in split_by_item(md, form):
        name = f"{base}_item{label}" + (f"-{title}" if title else "") + ".md"
        (d / name).write_text(body, encoding="utf-8")
        names.append(name)
    return names


def resplit_flat(root: Path, ticker: str) -> None:
    """Re-run the Item split on an existing --flat corpus. No network."""
    d = root / ticker
    mpath = d / "manifest.json"
    if not mpath.exists():
        print(f"  {ticker}: no manifest.json in {d}; --resplit needs a --flat corpus")
        return
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    for row in manifest.get("filings", []):
        if not row.get("form", "").startswith(("10-K", "10-Q")) or not row.get("file"):
            continue
        main_path = d / row["file"]
        if not main_path.exists():
            print(f"    missing {row['file']}")
            continue
        md = main_path.read_text(encoding="utf-8")
        names = write_splits(d, main_path.stem, md, row["form"])
        row["split_files"] = names
        dups = [n for n in names if SPLIT_DUP_RE.search(n)]
        flag = "" if not dups or row["form"].startswith("10-Q") else \
            f"  [!] repeated headings still present: {dups}"
        print(f"    {row['file']}: {len(names)} sections{flag}")
    manifest["resplit_at"] = datetime.now(timezone.utc).isoformat()
    manifest["splitter_version"] = SPLITTER_VERSION
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


SPLIT_DUP_RE = re.compile(r"_item\d+[A-Z]?-\d+(?:-|\.md$)")
SPLITTER_VERSION = "canon-align/1"


def flat_name(filing: Filing, suffix: str = "") -> str:
    """Self-describing filename for --flat mode: sorts sensibly, says what it
    is at a glance, and survives being dragged into an upload dialog."""
    t = filing.ticker
    form = filing.form.replace(" ", "-").replace("/", "")
    if filing.form.startswith("10-K"):
        stem = f"{t}_10-K_FY{filing.report_date[:4] or filing.filing_date[:4]}"
    elif filing.form.startswith("10-Q"):
        stem = f"{t}_10-Q_{filing.report_date or filing.filing_date}"
    elif filing.form.startswith("8-K"):
        items = filing.items.replace(",", "+").replace(" ", "")
        stem = f"{t}_8-K_{filing.filing_date}_i{items}"
    elif "14A" in filing.form:
        stem = f"{t}_{form}_{filing.filing_date[:4]}"
    else:
        stem = f"{t}_{form}_{filing.filing_date}"
    return f"{stem}{suffix}.md"


def filing_dir(root: Path, filing: Filing) -> Path:
    t = filing.ticker
    if filing.form.startswith("10-K"):
        sub = Path("10-K") / (filing.report_date[:4] or filing.filing_date[:4])
    elif filing.form.startswith("10-Q"):
        sub = Path("10-Q") / (filing.report_date or filing.filing_date)
    elif filing.form.startswith("8-K"):
        items = slug(filing.items.replace(",", "_"))
        sub = Path("8-K") / f"{filing.filing_date}-{items}"
    elif filing.form in ("DEF 14A", "DEFR14A"):
        # The proxy itself. Kept separate from DEFA14A below, because both
        # contain "14A" and are often filed the same day — routing them to one
        # directory silently drops the real proxy as already-cached.
        sub = Path("DEF14A") / (filing.filing_date[:4])
    elif "14A" in filing.form:
        # Additional soliciting material: usually a one-page press release.
        sub = Path("DEFA14A") / filing.filing_date
    else:
        sub = Path(slug(filing.form)) / filing.filing_date
    return root / t / sub


def write_meta(path: Path, filing: Filing, extra: dict):
    meta = asdict(filing)
    meta.pop("cik", None)
    meta.update({
        "cik": filing.cik,
        "source_url": filing.primary_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "converter_version": CONVERTER_VERSION,
        **extra,
    })
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


CONVERTER_VERSION = "edgar_fetch/1.7"


def write_manifest(root: Path, ticker: str, cik: str, rows: list[dict]):
    """In flat mode, one manifest replaces the per-filing meta.json files.
    Accession numbers still need a home — they are what makes a figure in the
    analysis traceable back to a source document."""
    d = root / ticker
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ticker": ticker,
        "cik": cik,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "converter_version": CONVERTER_VERSION,
        "filings": rows,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                     encoding="utf-8")

    lines = [f"# {ticker} filing corpus", "",
             f"CIK {cik} · generated {date.today().isoformat()}", "",
             "| File | Form | Period | Filed | Items | Accession |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(
            f"| `{r.get('file','')}` | {r.get('form','')} | "
            f"{r.get('report_date','') or '—'} | {r.get('filing_date','')} | "
            f"{r.get('items','') or '—'} | {r.get('accession','')} |"
        )
    (d / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch_document_filing(client: SecClient, filing: Filing, root: Path,
                          force: bool, flat: bool = False,
                          split: bool = False) -> dict:
    """Handle 10-K, 10-Q, 8-K, DEF 14A — anything that becomes markdown.

    In flat mode everything for a ticker lands in one directory with
    self-describing filenames, so an analysis run is a multi-select in a file
    dialog rather than a tour of twenty subfolders. Provenance moves from
    per-filing meta.json files into a single manifest.json."""
    if flat:
        d = root / filing.ticker
        main_path = d / flat_name(filing)
    else:
        d = filing_dir(root, filing)
        main_path = d / "source.md"

    marker = main_path if flat else (d / "meta.json")
    if marker.exists() and not force:
        return {"status": "cached", "dir": str(d), "file": main_path.name}
    d.mkdir(parents=True, exist_ok=True)

    html = client.get(filing.primary_url)
    if html is None:
        return {"status": "missing_primary", "dir": str(d)}

    md, report = html_to_markdown(html)
    main_path.write_text(md, encoding="utf-8")

    split_files = []
    if split and filing.form.startswith(("10-K", "10-Q")):
        split_files = write_splits(d, main_path.stem, md, filing.form)
        report["sections"] = len(split_files)

    exhibits = []
    if filing.form.startswith("8-K"):
        for name, url in find_exhibits(client, filing):
            ex_html = client.get(url)
            if not ex_html:
                continue
            ex_md, ex_rep = html_to_markdown(ex_html)
            label = slug(name.rsplit(".", 1)[0]).upper()
            if flat:
                ex_path = d / flat_name(filing, suffix=f"_{label}")
            else:
                ex_path = d / (slug(name.rsplit(".", 1)[0]) + ".md")
            ex_path.write_text(ex_md, encoding="utf-8")
            exhibits.append({"exhibit": name, "file": ex_path.name, **ex_rep})

    if not flat:
        write_meta(d, filing, {"conversion": report, "exhibits": exhibits})
    return {"status": "ok", "dir": str(d), "file": main_path.name,
            **report, "exhibit_count": len(exhibits),
            "exhibit_files": [e["file"] for e in exhibits],
            "split_files": split_files}


def fetch_ownership_filings(client: SecClient, filings: list[Filing],
                            root: Path, ticker: str) -> dict:
    """Form 4/5 and 144 become CSV rows, never documents. Three hundred
    filings is one spreadsheet, not three hundred things to read."""
    form4_rows, form144_rows = [], []
    errors = 0

    for f in filings:
        url = find_ownership_xml(client, f)
        if not url:
            errors += 1
            continue
        xml = client.get(url)
        if not xml:
            errors += 1
            continue
        try:
            if f.form.startswith("144"):
                form144_rows.extend(parse_form144(xml, f.accession, f.filing_date))
            else:
                form4_rows.extend(parse_form4(xml, f.accession, f.filing_date))
        except Exception as e:  # noqa: BLE001 - one bad filing shouldn't kill the run
            print(f"    ! parse error {f.accession}: {e}", file=sys.stderr)
            errors += 1

    out = root / ticker
    out.mkdir(parents=True, exist_ok=True)
    written = {}

    if form4_rows:
        p = out / "form4_transactions.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(form4_rows[0].keys()))
            w.writeheader()
            w.writerows(form4_rows)
        written["form4_csv"] = str(p)
        written["form4_rows"] = len(form4_rows)

        signal = [r for r in form4_rows if r["is_signal_code"]]
        p2 = out / "form4_signal_only.csv"
        with p2.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(form4_rows[0].keys()))
            w.writeheader()
            w.writerows(signal)
        written["form4_signal_rows"] = len(signal)
        written["form4_signal_csv"] = str(p2)

        summary_path = out / "form4_summary.md"
        summary_path.write_text(summarize_form4(signal), encoding="utf-8")
        written["form4_summary"] = str(summary_path)

    if form144_rows:
        p = out / "form144_notices.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(form144_rows[0].keys()))
            w.writeheader()
            w.writerows(form144_rows)
        written["form144_csv"] = str(p)
        written["form144_rows"] = len(form144_rows)

    written["parse_errors"] = errors
    return written


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", required=True,
                    help="Comma-separated, e.g. GTLB,TEAM,FROG")
    ap.add_argument("--email", default=None,
                    help="Contact email for the SEC User-Agent (required)")
    ap.add_argument("--years", type=float, default=2.0,
                    help="Trailing window in years (default 2)")
    ap.add_argument("--out", default="./corpus", help="Output root")
    ap.add_argument("--forms", default="10-K,10-Q,8-K,DEF 14A,4,144",
                    help="Which form groups to fetch")
    ap.add_argument("--items", default=",".join(sorted(DEFAULT_8K_ITEMS)),
                    help="8-K items to keep")
    ap.add_argument("--all-8k", action="store_true",
                    help="Disable 8-K item filtering (not recommended)")
    ap.add_argument("--flat", action="store_true",
                    help="One folder per ticker with self-describing "
                         "filenames, for easy multi-select upload")
    ap.add_argument("--split", action="store_true",
                    help="Also write per-Item files for 10-K/10-Q, so a full "
                         "filing need not be uploaded to read one section")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if cached")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be fetched, download nothing")
    ap.add_argument("--resplit", action="store_true",
                    help="Re-split existing 10-K/10-Q files in a --flat corpus "
                         "into Items. No downloads; --email not needed")
    args = ap.parse_args()

    if args.resplit:
        root = Path(args.out).expanduser().resolve()
        for t in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
            print(f"=== {t} (resplit) ===")
            resplit_flat(root, t)
        return

    email = args.email
    if not email:
        print("ERROR: --email is required. The SEC blocks requests without a "
              "real contact address in the User-Agent.", file=sys.stderr)
        sys.exit(1)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    groups = [g.strip() for g in args.forms.split(",") if g.strip()]
    wanted_items = set() if args.all_8k else {i.strip() for i in args.items.split(",")}

    wanted_forms: set[str] = set()
    for g in groups:
        wanted_forms.update(FORM_GROUPS.get(g, [g]))

    since = date.today() - timedelta(days=int(args.years * 365.25))
    root = Path(args.out).expanduser().resolve()

    client = SecClient(email)
    print(f"Window: {since.isoformat()} -> {date.today().isoformat()}")
    print(f"Forms:  {sorted(wanted_forms)}")
    print(f"Output: {root}\n")

    ciks = resolve_ciks(client, tickers)
    summary = {}

    for ticker, cik in ciks.items():
        print(f"=== {ticker} (CIK {cik}) ===")
        filings = list_filings(client, ticker, cik, since, wanted_forms)

        docs, ownership = [], []
        skipped_8k = 0
        for f in filings:
            if f.form.startswith("8-K") and wanted_items:
                if not keep_8k(f, wanted_items):
                    skipped_8k += 1
                    continue
            if f.form in ("4", "4/A", "5", "5/A", "144", "144/A"):
                ownership.append(f)
            else:
                docs.append(f)

        counts: dict[str, int] = {}
        for f in docs:
            counts[f.form] = counts.get(f.form, 0) + 1
        own_counts: dict[str, int] = {}
        for f in ownership:
            own_counts[f.form] = own_counts.get(f.form, 0) + 1

        print(f"  documents: {counts}")
        print(f"  ownership: {own_counts}")
        if skipped_8k:
            print(f"  8-Ks filtered out as non-material: {skipped_8k}")

        if args.dry_run:
            summary[ticker] = {"documents": counts, "ownership": own_counts,
                               "skipped_8k": skipped_8k}
            print()
            continue

        results = []
        manifest_rows = []
        for f in docs:
            r = fetch_document_filing(client, f, root, args.force, args.flat,
                                      args.split)
            results.append({"form": f.form, "date": f.filing_date,
                            "items": f.items, **r})
            manifest_rows.append({
                "file": r.get("file", ""),
                "form": f.form,
                "report_date": f.report_date,
                "filing_date": f.filing_date,
                "items": f.items,
                "accession": f.accession,
                "source_url": f.primary_url,
                "exhibit_files": r.get("exhibit_files", []),
                "split_files": r.get("split_files", []),
            })
            flag = ""
            # DEFA14A is additional soliciting material — usually a one-page
            # letter with no tables. Flagging it every run trains you to
            # ignore the warning, which defeats the point of having one.
            if (r.get("status") == "ok" and r.get("tables_kept", 0) == 0
                    and not f.form.startswith(("DEFA", "8-K"))):
                flag = "  [!] no tables extracted — check manually"
            if f.form.startswith("8-K") and r.get("status") == "ok" \
                    and r.get("exhibit_count", 0) == 0 and "2.02" in (f.items or ""):
                flag = "  [!] Item 2.02 with no EX-99 — earnings release missing"
            extras = []
            if r.get("exhibit_count"):
                extras.append(f"ex={r['exhibit_count']}")
            if r.get("sections"):
                extras.append(f"sections={r['sections']}")
            extra = ("  " + " ".join(extras)) if extras else ""
            print(f"    {f.form:10s} {f.filing_date}  {r['status']}"
                  f"  tables={r.get('tables_kept', '-')}{extra}{flag}")

        own = {}
        if ownership:
            print(f"    parsing {len(ownership)} ownership filings...")
            own = fetch_ownership_filings(client, ownership, root, ticker)
            if "form4_rows" in own:
                print(f"    Form 4: {own['form4_rows']} transactions, "
                      f"{own['form4_signal_rows']} open-market (P/S)")
            if "form144_rows" in own:
                print(f"    Form 144: {own['form144_rows']} notices")

        if args.flat:
            write_manifest(root, ticker, cik, manifest_rows)
            print(f"    manifest: {root / ticker / 'manifest.md'}")

        summary[ticker] = {"documents": results, "ownership": own,
                           "skipped_8k": skipped_8k, "cik": cik}
        print()

    if not args.dry_run:
        rp = root / "fetch_report.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Report: {rp}")


if __name__ == "__main__":
    main()
