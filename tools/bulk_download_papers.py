#!/usr/bin/env python3
"""Bulk-download open-access PDFs from a bibliography file using DOIs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

BIB_FILE = Path("/Users/soccerment/Downloads/100_essential_papers_titles_authors_doi.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "papers"
CSV_FILE = OUTPUT_DIR / "bibliography.csv"
LOG_FILE = OUTPUT_DIR / "download_log.json"

UNPAYWALL_EMAIL = "research@example.com"
REQUEST_DELAY = 0.5
USER_AGENT = "SportsScienceRAG/1.0 (mailto:research@example.com)"


@dataclass
class Paper:
    id: int
    title: str
    authors: str
    doi: str
    doi_resolved: str = ""
    status: str = "pending"
    pdf_url: str = ""
    source: str = ""
    filename: str = ""
    error: str = ""


def http_get(
    url: str,
    params: dict[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> bytes:
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("http_get failed without exception")


def http_get_json(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    raw = http_get(url, params=params)
    return json.loads(raw.decode("utf-8"))


def parse_bibliography(path: Path) -> list[Paper]:
    papers: list[Paper] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\.\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.*)$", line)
        if not match:
            continue
        paper_id = int(match.group(1))
        title = match.group(2).strip()
        authors = match.group(3).strip()
        doi_field = match.group(4).strip()
        doi = ""
        if doi_field:
            doi_match = re.search(r"doi\.org/(.+)$", doi_field, re.IGNORECASE)
            if doi_match:
                doi = doi_match.group(1).strip().rstrip(".")
        papers.append(Paper(id=paper_id, title=title, authors=authors, doi=doi))
    return papers


def first_author_slug(authors: str) -> str:
    if not authors:
        return "unknown"
    first = re.split(r"[,;&]", authors)[0].strip()
    first = re.sub(r"[^a-zA-Z0-9]+", "_", first).strip("_").lower()
    return first or "unknown"


def safe_filename(paper: Paper) -> str:
    return f"{paper.id:03d}_{first_author_slug(paper.authors)}.pdf"


def normalize_doi(doi: str) -> str:
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi.rstrip("/.")


def is_complete_doi(doi: str) -> bool:
    if not doi:
        return False
    if doi.endswith("."):
        return False
    if doi in {"10", "10."}:
        return False
    if not re.match(r"^10\.\d{4,9}/[^\s/].+$", doi):
        return False
    suffix = doi.split("/", 1)[1]
    # Truncated list entries often stop at journal abbreviations (e.g. 10.1152/jappl).
    if len(suffix) < 12:
        return False
    if re.match(r"^j[a-z]+$", suffix):
        return False
    if re.match(r"^jphysiol\.\d{4}$", suffix):
        return False
    if re.match(r"^jappl$", suffix):
        return False
    return True


def crossref_search(paper: Paper) -> str:
    query = f"{paper.title} {paper.authors.split(',')[0]}"
    data = http_get_json(
        "https://api.crossref.org/works",
        params={"query.bibliographic": query, "rows": "5"},
    )
    items = data.get("message", {}).get("items", [])
    title_lower = paper.title.lower()
    for item in items:
        item_titles = [t.lower() for t in item.get("title", [])]
        if not item_titles:
            continue
        if title_lower[:40] in item_titles[0] or item_titles[0][:40] in title_lower:
            doi = item.get("DOI", "")
            if doi:
                return normalize_doi(doi)
    if items:
        doi = items[0].get("DOI", "")
        if doi:
            return normalize_doi(doi)
    return ""


def resolve_dois(papers: list[Paper]) -> None:
    for paper in papers:
        if is_complete_doi(paper.doi):
            paper.doi_resolved = normalize_doi(paper.doi)
            continue
        if is_complete_doi(paper.doi_resolved):
            continue
        try:
            resolved = crossref_search(paper)
            paper.doi_resolved = resolved
            if resolved:
                paper.status = "doi_resolved"
            else:
                paper.status = "doi_unresolved"
                paper.error = "CrossRef lookup failed"
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            paper.status = "doi_unresolved"
            paper.error = f"CrossRef error: {exc}"
        time.sleep(REQUEST_DELAY)


def unpaywall_pdf(doi: str) -> tuple[str, str]:
    try:
        data = http_get_json(
            f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}",
            params={"email": UNPAYWALL_EMAIL},
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "", ""
        raise
    best = data.get("best_oa_location") or {}
    pdf_url = best.get("url_for_pdf") or best.get("url", "")
    if pdf_url and pdf_url.endswith(".pdf"):
        return pdf_url, "unpaywall"
    for location in data.get("oa_locations", []):
        pdf_url = location.get("url_for_pdf") or ""
        if pdf_url:
            return pdf_url, "unpaywall"
        host = urlparse(location.get("url", "")).netloc
        if "pmc.ncbi.nlm.nih.gov" in host or "europepmc.org" in host:
            return location["url"], "unpaywall_pmc"
    return "", ""


def europe_pmc_pdf(doi: str) -> tuple[str, str]:
    data = http_get_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": f"DOI:{doi}", "format": "json", "pageSize": "1"},
    )
    results = data.get("resultList", {}).get("result", [])
    if not results:
        return "", ""
    record = results[0]
    pmcid = record.get("pmcid")
    if not pmcid:
        return "", ""
    pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
    return pdf_url, "europe_pmc"


def semantic_scholar_pdf(doi: str) -> tuple[str, str]:
    try:
        data = http_get_json(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}",
            params={"fields": "openAccessPdf,isOpenAccess"},
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "", ""
        raise
    oa = data.get("openAccessPdf") or {}
    pdf_url = oa.get("url", "")
    if pdf_url:
        return pdf_url, "semantic_scholar"
    return "", ""


def normalize_pdf_url(url: str, doi: str = "") -> list[str]:
    """Return candidate PDF URLs, best first."""
    candidates: list[str] = []

    if doi.startswith("10.1371/"):
        candidates.append(
            f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable"
        )

    if doi.startswith("10.1186/"):
        candidates.append(f"https://link.springer.com/content/pdf/{doi}.pdf")

    if doi.startswith("10.3390/"):
        candidates.append(f"https://mdpi.com/res/pdf/{doi.replace('/', '_')}")

    if doi.startswith("10.3389/"):
        candidates.append(f"https://www.frontiersin.org/articles/{doi}/pdf")

    if doi.startswith("10.1136/bjsm"):
        candidates.append(f"https://bjsm.bmj.com/content/bjsports/{doi.split('.')[-1]}.full.pdf")

    pmc_match = re.search(r"(PMC\d+)", url or "", re.IGNORECASE)
    if pmc_match:
        pmcid = pmc_match.group(1).upper()
        candidates.extend([
            f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
        ])

    if url:
        candidates.append(url)

    if doi:
        candidates.append(f"https://doi.org/{doi}")

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def find_pdf_url(doi: str) -> tuple[str, str]:
    for finder in (unpaywall_pdf, europe_pmc_pdf, semantic_scholar_pdf):
        try:
            pdf_url, source = finder(doi)
            if pdf_url:
                return pdf_url, source
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            continue
        time.sleep(REQUEST_DELAY)
    return "", ""


def download_pdf(url: str, dest: Path, doi: str = "") -> bool:
    for candidate in normalize_pdf_url(url, doi):
        try:
            content = http_get(candidate, timeout=60.0, retries=3)
            if content.startswith(b"%PDF"):
                dest.write_bytes(content)
                if dest.stat().st_size > 1024:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            continue
    return False


def save_csv(papers: list[Paper], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "title",
                "authors",
                "doi",
                "doi_resolved",
                "status",
                "pdf_url",
                "source",
                "filename",
                "error",
            ],
        )
        writer.writeheader()
        for paper in papers:
            writer.writerow(asdict(paper))


def load_papers_from_csv(path: Path) -> list[Paper]:
    papers: list[Paper] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            papers.append(Paper(**{k: row.get(k, "") for k in Paper.__dataclass_fields__}))
            papers[-1].id = int(papers[-1].id)
    return papers


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk-download open-access PDFs from bibliography")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed downloads from existing CSV")
    args = parser.parse_args()

    if not BIB_FILE.exists() and not (args.retry_failed and CSV_FILE.exists()):
        print(f"Bibliography not found: {BIB_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.retry_failed and CSV_FILE.exists():
        papers = load_papers_from_csv(CSV_FILE)
        print(f"Loaded {len(papers)} papers from CSV for retry.")
    else:
        papers = parse_bibliography(BIB_FILE)
        print(f"Parsed {len(papers)} papers from bibliography.")
        print("Resolving incomplete DOIs via CrossRef...")
        resolve_dois(papers)
        resolved_count = sum(1 for p in papers if p.doi_resolved)
        print(f"Resolved DOIs: {resolved_count}/{len(papers)}")

    save_csv(papers, CSV_FILE)
    print(f"Saved CSV: {CSV_FILE}")

    downloaded = 0
    skipped = 0
    failed = 0

    for paper in papers:
        paper.filename = safe_filename(paper)
        dest = OUTPUT_DIR / paper.filename
        if paper.status == "downloaded" and dest.exists() and dest.stat().st_size > 1024:
            paper.status = "already_exists"
            skipped += 1
            continue

        if args.retry_failed and paper.status == "downloaded":
            skipped += 1
            continue

        if not paper.doi_resolved:
            paper.status = "no_doi"
            failed += 1
            continue

        pdf_url, source = find_pdf_url(paper.doi_resolved)
        existing_url = paper.pdf_url if paper.pdf_url else pdf_url
        if pdf_url:
            paper.pdf_url = pdf_url
            paper.source = source

        try:
            if download_pdf(existing_url or "", dest, paper.doi_resolved):
                paper.status = "downloaded"
                downloaded += 1
                print(f"  [{paper.id:03d}] OK  {paper.filename} ({source})")
            else:
                paper.status = "no_open_access"
                paper.error = "No OA PDF found (may need institutional access)"
                if dest.exists():
                    dest.unlink()
                failed += 1
        except Exception as exc:
            paper.status = "download_failed"
            paper.error = str(exc)
            failed += 1
        time.sleep(REQUEST_DELAY)
        save_csv(papers, CSV_FILE)

    save_csv(papers, CSV_FILE)
    LOG_FILE.write_text(
        json.dumps([asdict(p) for p in papers], indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print(f"Downloaded : {downloaded}")
    print(f"Skipped    : {skipped} (already on disk)")
    print(f"Failed     : {failed}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Log        : {LOG_FILE}")

    manual = [p for p in papers if p.status in {"no_open_access", "no_doi", "doi_unresolved"}]
    if manual:
        print()
        print(f"Papers needing manual access ({len(manual)}):")
        for paper in manual:
            doi_str = paper.doi_resolved or paper.doi or "?"
            print(f"  [{paper.id:03d}] {paper.title[:60]}... | DOI: {doi_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
