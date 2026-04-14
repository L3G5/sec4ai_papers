from __future__ import annotations

import argparse
import logging
import re
import shutil
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import date, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import pandas as pd
from tqdm.auto import tqdm

from arxiv_priority_predictor import ArxivPriorityPredictor


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_EXPORT_API_URL = "https://export.arxiv.org/api/query"
ARXIV_RECENT_URL_TEMPLATE = "https://arxiv.org/list/{category}/recent?show=2000"
DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CR"]
LOG_DIR = Path("logs")
LODS_DIR = Path("lods")
SECTION_RE = re.compile(
    r"<h3>\s*([^<]+?)\s*\(showing\s+(\d+)\s+of\s+(\d+)\s+entries\s*\)\s*</h3>(.*?)(?=<h3>|</dl>)",
    re.S,
)
ID_RE = re.compile(r'href\s*=\s*"/abs/([0-9]+\.[0-9]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a day's arXiv papers using recent pages for IDs and the export API for metadata."
    )
    parser.add_argument(
        "--day",
        default=None,
        help="Day to fetch in YYYY-MM-DD format. Default: latest available day from the recent pages.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=DEFAULT_CATEGORIES,
        help="Categories to include. Default: cs.AI cs.LG cs.CL cs.CR",
    )
    parser.add_argument(
        "--model",
        default="artifacts/arxiv_priority_predictor_2511_abstract_authors_surnames.joblib",
        help="Saved predictor bundle.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/arxiv_api_daily_scores.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--api-batch-size",
        type=int,
        default=100,
        help="Number of arXiv IDs per export API id_list request.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
        help="Pause between export API calls.",
    )
    parser.add_argument("--log-max-bytes", type=int, default=2_000_000)
    parser.add_argument("--log-backup-count", type=int, default=5)
    return parser.parse_args()


def parse_day(day_text: str) -> date:
    try:
        return datetime.strptime(day_text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid --day value: {day_text}. Expected YYYY-MM-DD.") from exc

import requests


def fetch_url(url: str) -> str:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    request = urllib.request.Request(
        url,
        headers=headers,
    )
    time.sleep(2)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def setup_logger(log_max_bytes: int, log_backup_count: int) -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"score_arxiv_api_day_{timestamp}.log"

    logger = logging.getLogger("score_arxiv_api_day")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = RotatingFileHandler(log_path, maxBytes=log_max_bytes, backupCount=log_backup_count, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def parse_recent_sections(html: str) -> list[tuple[date, int, list[str]]]:
    sections: list[tuple[date, int, list[str]]] = []
    for label, shown_count, _, block in SECTION_RE.findall(html):
        day_value = datetime.strptime(" ".join(label.split()), "%a, %d %b %Y").date()
        ids = list(OrderedDict.fromkeys(ID_RE.findall(block)))
        sections.append((day_value, int(shown_count), ids))
    return sections


def recent_ids_for_category(category: str, logger: logging.Logger) -> list[tuple[date, int, list[str]]]:
    url = ARXIV_RECENT_URL_TEMPLATE.format(category=category)
    logger.info("Fetching recent page for %s: %s", category, url)
    html = fetch_url(url)
    sections = parse_recent_sections(html)
    if not sections:
        raise SystemExit(f"Could not parse recent page for {category}")
    return sections


def resolve_target_day(categories: list[str], requested_day: str | None, logger: logging.Logger) -> tuple[date, dict[str, list[str]]]:
    per_category_sections = {category: recent_ids_for_category(category, logger) for category in categories}
    if requested_day is None:
        target_day = per_category_sections[categories[0]][0][0]
        logger.info("Resolved default day from %s recent page: %s", categories[0], target_day.isoformat())
    else:
        target_day = parse_day(requested_day)

    per_category_ids: dict[str, list[str]] = {}
    for category, sections in per_category_sections.items():
        ids: list[str] = []
        shown_count = 0
        for section_day, section_shown_count, section_ids in sections:
            if section_day == target_day:
                ids = section_ids
                shown_count = section_shown_count
                break
        per_category_ids[category] = ids
        logger.info(
            "Recent page count for %s on %s: shown=%s parsed_ids=%s",
            category,
            target_day.isoformat(),
            shown_count,
            len(ids),
        )
    return target_day, per_category_ids


def dedupe_ids(per_category_ids: dict[str, list[str]]) -> list[str]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for ids in per_category_ids.values():
        for paper_id in ids:
            ordered.setdefault(paper_id, None)
    return list(ordered.keys())


def parse_export_entry(entry: ET.Element) -> dict[str, str]:
    authors = [
        author.findtext("atom:name", default="", namespaces=ATOM_NS).strip()
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    categories = [category.attrib.get("term", "").strip() for category in entry.findall("atom:category", ATOM_NS)]
    abs_url = ""
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("rel") == "alternate":
            abs_url = link.attrib.get("href", "").strip()
            break
    if not abs_url:
        abs_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
    paper_id = abs_url.rsplit("/", 1)[-1].split("v", 1)[0]
    return {
        "paper_id": paper_id,
        "url": abs_url.replace("http://", "https://"),
        "title": " ".join(entry.findtext("atom:title", default="", namespaces=ATOM_NS).split()),
        "abstract": " ".join(entry.findtext("atom:summary", default="", namespaces=ATOM_NS).split()),
        "authors": ", ".join(author for author in authors if author),
        "categories": " ".join(categories),
        "published": entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip(),
        "updated": entry.findtext("atom:updated", default="", namespaces=ATOM_NS).strip(),
    }


def fetch_metadata_for_ids(
    paper_ids: list[str],
    batch_size: int,
    pause_seconds: float,
    logger: logging.Logger,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    time.sleep(10)
    progress = tqdm(range(0, len(paper_ids), batch_size), desc="Fetching export API batches")
    for start in progress:
        batch_ids = paper_ids[start : start + batch_size]
        logger.info("Fetching export API batch %s-%s", start, start + len(batch_ids))
        url = ARXIV_EXPORT_API_URL + "?" + urllib.parse.urlencode(
            {"id_list": ",".join(batch_ids), "max_results": len(batch_ids)}
        )
        xml_text = fetch_url(url)
        root = ET.fromstring(xml_text)
        batch_rows = [parse_export_entry(entry) for entry in root.findall("atom:entry", ATOM_NS)]
        rows.extend(batch_rows)
        time.sleep(pause_seconds)
    return rows


def save_output(frame: pd.DataFrame, output_path: Path, day_value: date) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    LODS_DIR.mkdir(parents=True, exist_ok=True)
    lods_path = LODS_DIR / f"arxiv_api_scores_{day_value.isoformat()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    shutil.copy2(output_path, lods_path)
    return lods_path


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger(args.log_max_bytes, args.log_backup_count)
    logger.info("Starting daily arXiv scoring")
    logger.info("Requested day=%s Categories=%s Output=%s", args.day, args.categories, args.output)

    target_day, per_category_ids = resolve_target_day(args.categories, args.day, logger)
    union_ids = dedupe_ids(per_category_ids)
    logger.info("Resolved day=%s unique_ids=%s", target_day.isoformat(), len(union_ids))

    if not union_ids:
        output = pd.DataFrame(columns=["url", "abstract", "score"])
        lods_path = save_output(output, Path(args.output), target_day)
        logger.info("No IDs found on recent pages for %s", target_day.isoformat())
        logger.info("Saved empty parquet to %s", args.output)
        logger.info("Saved lods parquet to %s", lods_path)
        print(f"No recent-page IDs found for {target_day.isoformat()}")
        print(f"Saved empty parquet to {args.output}")
        print(f"Saved lods parquet to {lods_path}")
        print(f"Log file: {log_path}")
        return

    api_rows = fetch_metadata_for_ids(union_ids, args.api_batch_size, args.pause_seconds, logger)
    api_by_id = {row["paper_id"]: row for row in api_rows}
    missing_ids = [paper_id for paper_id in union_ids if paper_id not in api_by_id]
    logger.info("Recent-page unique IDs=%s, export API returned=%s, missing=%s", len(union_ids), len(api_rows), len(missing_ids))
    if missing_ids:
        logger.warning("Missing IDs from export API: %s", ", ".join(missing_ids[:20]))

    ordered_rows = [api_by_id[paper_id] for paper_id in union_ids if paper_id in api_by_id]
    papers = pd.DataFrame(ordered_rows)
    if papers.empty:
        output = pd.DataFrame(columns=["url", "abstract", "score"])
        lods_path = save_output(output, Path(args.output), target_day)
        logger.info("Export API returned no metadata rows for %s", target_day.isoformat())
        logger.info("Saved empty parquet to %s", args.output)
        logger.info("Saved lods parquet to %s", lods_path)
        print(f"No export API metadata rows for {target_day.isoformat()}")
        print(f"Saved empty parquet to {args.output}")
        print(f"Saved lods parquet to {lods_path}")
        print(f"Log file: {log_path}")
        return

    predictor: ArxivPriorityPredictor = joblib.load(args.model)
    logger.info("Loaded model bundle from %s", args.model)
    scored = predictor.score(papers)
    result = scored.loc[:, ["url", "title", "authors", "abstract", "score"]].copy()
    lods_path = save_output(result, Path(args.output), target_day)

    print(f"Day: {target_day.isoformat()}")
    print(f"Categories: {', '.join(args.categories)}")
    for category in args.categories:
        print(f"{category}: {len(per_category_ids[category])} recent-page IDs")
    print(f"Unique IDs across categories: {len(union_ids)}")
    print(f"Metadata rows returned by export API: {len(api_rows)}")
    print(f"Saved parquet to {args.output}")
    print(f"Saved lods parquet to {lods_path}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
