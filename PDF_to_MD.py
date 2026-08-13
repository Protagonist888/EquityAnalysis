#!/usr/bin/env python3
"""
Convert all .pdf files in a folder to .md files.
Tuned for SEC filings (10-K, 10-Q, 8-K, etc.):
  - Strips repeated headers/footers (company name, form type, page numbers)
  - Converts "Item 1A. Risk Factors" style lines into Markdown headers
  - Uses table settings better suited to financial statement tables

Usage:
    python pdf_to_md.py /path/to/folder

Skips any .pdf file that already has a matching .md file in the same folder.
Requires: pip install pdfplumber
"""

import re
import sys
from collections import Counter
from pathlib import Path
import pdfplumber

# Matches lines like "Item 1A. Risk Factors" or "ITEM 7 - MD&A"
ITEM_HEADING_RE = re.compile(r"^\s*ITEM\s+\d+[A-Z]?\.?\s*[-–—]?\s*.+$", re.IGNORECASE)

# Financial-statement-friendly table extraction settings
TABLE_SETTINGS = {
    "vertical_strategy": "lines_strict",
    "horizontal_strategy": "lines_strict",
    "snap_tolerance": 4,
    "join_tolerance": 4,
}


def table_to_md(table):
    """Convert a pdfplumber table (list of rows) into a Markdown table."""
    if not table or len(table) < 2:
        return ""

    rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table]
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def find_repeated_lines(pages_text, min_page_count):
    """Find lines (e.g. headers/footers) that repeat across most pages, to strip them."""
    line_counts = Counter()
    for text in pages_text:
        seen_this_page = set()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and stripped not in seen_this_page:
                line_counts[stripped] += 1
                seen_this_page.add(stripped)
    return {line for line, count in line_counts.items() if count >= min_page_count}


def format_line(line):
    stripped = line.strip()
    if ITEM_HEADING_RE.match(stripped):
        return f"## {stripped}"
    return line


# Font-encoding failures often show up as literal "(cid:123)" placeholders
# instead of real characters — a sign the PDF's font mapping is broken.
CID_RE = re.compile(r"\(cid:\d+\)")


def convert_pdf_to_md(pdf_path: Path, verbose=True):
    """Returns (markdown_text, quality_report_dict)."""
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        pages_text = []
        pages_tables = []

        for i, page in enumerate(pdf.pages, start=1):
            if verbose and num_pages > 20 and i % 20 == 0:
                print(f"  ...page {i}/{num_pages}")
            pages_text.append(page.extract_text() or "")
            try:
                pages_tables.append(page.extract_tables(TABLE_SETTINGS))
            except Exception:
                pages_tables.append([])

        # Lines repeating on most pages are almost certainly headers/footers
        # (company name, form type, ticker, page numbers) — strip them.
        threshold = max(3, int(num_pages * 0.6))
        repeated = find_repeated_lines(pages_text, threshold) if num_pages >= 3 else set()

        # --- Quality signals, gathered as we go ---
        char_counts = [len(t) for t in pages_text]
        avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0
        low_text_pages = [
            i for i, c in enumerate(char_counts, start=1)
            if c < avg_chars * 0.15 and avg_chars > 200  # ignore genuinely short docs
        ]
        cid_pages = [
            i for i, t in enumerate(pages_text, start=1) if CID_RE.search(t)
        ]
        total_tables = sum(len(t) for t in pages_tables)

        lines = []
        for i, (text, tables) in enumerate(zip(pages_text, pages_tables), start=1):
            page_lines = [
                format_line(l) for l in text.splitlines()
                if l.strip() not in repeated
            ]
            cleaned = "\n".join(page_lines).strip()

            if cleaned:
                lines.append(cleaned)

            for table in tables:
                md_table = table_to_md(table)
                if md_table:
                    lines.append("")
                    lines.append(md_table)

            lines.append("")  # spacer between pages

    markdown = "\n\n".join(lines).strip() + "\n"
    report = {
        "num_pages": num_pages,
        "total_tables": total_tables,
        "low_text_pages": low_text_pages,
        "cid_encoding_pages": cid_pages,
    }
    return markdown, report


def main():
    if len(sys.argv) != 2:
        print("Usage: python pdf_to_md.py /path/to/folder")
        sys.exit(1)

    folder = Path(sys.argv[1]).expanduser().resolve()
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(1)

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No .pdf files found in {folder}")
        return

    converted = 0
    skipped = 0
    report_lines = []

    for pdf_path in pdf_files:
        md_path = pdf_path.with_suffix(".md")

        if md_path.exists():
            print(f"Skipping (already exists): {md_path.name}")
            skipped += 1
            continue

        try:
            print(f"Processing: {pdf_path.name}")
            md_content, report = convert_pdf_to_md(pdf_path)

            warnings = []
            if not md_content.strip():
                warnings.append("NO EXTRACTABLE TEXT — likely scanned/image-based PDF. Needs OCR, not this script.")
            if report["cid_encoding_pages"]:
                warnings.append(
                    f"Possible font-encoding garbling on page(s): {report['cid_encoding_pages']} "
                    "(literal '(cid:N)' placeholders found instead of real characters)"
                )
            if report["low_text_pages"]:
                warnings.append(
                    f"Unusually little text extracted on page(s): {report['low_text_pages']} "
                    "(could be a chart/image, or an extraction miss — worth a manual look)"
                )
            if report["total_tables"] == 0:
                warnings.append("No tables detected in the whole document — check manually if this filing should have financial tables.")

            md_path.write_text(md_content, encoding="utf-8")
            print(f"Converted: {pdf_path.name} -> {md_path.name}")
            converted += 1

            report_lines.append(f"## {pdf_path.name}")
            report_lines.append(f"Pages: {report['num_pages']} | Tables found: {report['total_tables']}")
            if warnings:
                for w in warnings:
                    report_lines.append(f"  ⚠ {w}")
            else:
                report_lines.append("  No issues flagged.")
            report_lines.append("")

        except Exception as e:
            print(f"Error converting {pdf_path.name}: {e}")
            report_lines.append(f"## {pdf_path.name}")
            report_lines.append(f"  ⚠ FAILED TO CONVERT: {e}")
            report_lines.append("")

    if report_lines:
        report_path = folder / "conversion_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\nQuality report written to: {report_path.name}")

    print(f"Done. Converted: {converted}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
