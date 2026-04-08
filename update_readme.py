from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


START_MARKER = "<!-- README_LATEST_PAPERS_START -->"
END_MARKER = "<!-- README_LATEST_PAPERS_END -->"
DAY_HEADER_RE = re.compile(r"^### (\d{4}-\d{2}-\d{2})$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the README section with the latest high-score papers.")
    parser.add_argument("input_path", help="Parquet file with scored papers.")
    parser.add_argument("--readme", default="README.md", help="README path to update.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Include papers with score >= threshold.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of papers to render.")
    parser.add_argument("--day", default=None, help="Scoring day in YYYY-MM-DD format.")
    return parser.parse_args()


def render_day_block(frame: pd.DataFrame, *, threshold: float, limit: int, day: str | None) -> str:
    filtered = frame[frame["score"] >= threshold].sort_values("score", ascending=False).head(limit)
    heading = day or datetime.now(UTC).strftime("%Y-%m-%d")
    lines = [f"### {heading}", ""]
    if day:
        lines.append(f"Run date: `{day}`.")
    else:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"Run timestamp: `{stamp}`.")
    lines.append("")

    if filtered.empty:
        lines.append(f"No papers reached `{threshold:.0%}` in this run.")
    else:
        for row in filtered.itertuples(index=False):
            title = " ".join(str(row.title).split())
            authors = " ".join(str(getattr(row, "authors", "")).split())
            abstract = " ".join(str(getattr(row, "abstract", "")).split())
            score = float(row.score)
            url = str(row.url)
            lines.append(f"- [{title}]({url}) ({score:.1%})")
            if authors:
                lines.append(f"  Authors: {authors}")
            if abstract:
                lines.append(f"  {abstract}")
    return "\n".join(lines)


def parse_existing_history(section_body: str) -> list[tuple[str | None, str]]:
    section_body = section_body.strip()
    if not section_body:
        return []

    matches = list(DAY_HEADER_RE.finditer(section_body))
    if not matches:
        return [(None, section_body)]

    blocks: list[tuple[str | None, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section_body)
        blocks.append((match.group(1), section_body[start:end].strip()))
    return blocks


def build_history_section(
    readme_text: str,
    frame: pd.DataFrame,
    *,
    threshold: float,
    limit: int,
    day: str | None,
) -> str:
    start = readme_text.find(START_MARKER)
    end = readme_text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise SystemExit("README managed section markers were not found.")

    body = readme_text[start + len(START_MARKER) : end].strip()
    entries = parse_existing_history(body)
    new_day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    new_block = render_day_block(frame, threshold=threshold, limit=limit, day=new_day)

    filtered_entries = [(entry_day, block) for entry_day, block in entries if entry_day != new_day]
    blocks = [new_block] + [block for _, block in filtered_entries]
    lines = [
        START_MARKER,
        "## Papers Above 50%",
        "",
        "Archived by run date, newest first.",
        "",
        "\n\n".join(blocks).strip(),
        "",
        END_MARKER,
    ]
    return "\n".join(lines)


def replace_managed_section(readme_text: str, section_text: str) -> str:
    start = readme_text.find(START_MARKER)
    end = readme_text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise SystemExit("README managed section markers were not found.")
    end += len(END_MARKER)
    return readme_text[:start] + section_text + readme_text[end:]


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input_path)
    required = {"url", "score", "abstract"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Input parquet is missing required columns: {sorted(missing)}")
    for optional in ["title", "authors"]:
        if optional not in frame.columns:
            frame[optional] = ""

    readme_path = Path(args.readme)
    readme_text = readme_path.read_text(encoding="utf-8")
    section = build_history_section(readme_text, frame, threshold=args.threshold, limit=args.limit, day=args.day)
    updated = replace_managed_section(readme_text, section)
    readme_path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
