from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


START_MARKER = "<!-- README_LATEST_PAPERS_START -->"
END_MARKER = "<!-- README_LATEST_PAPERS_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the README section with the latest high-score papers.")
    parser.add_argument("input_path", help="Parquet file with scored papers.")
    parser.add_argument("--readme", default="README.md", help="README path to update.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Include papers with score >= threshold.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of papers to render.")
    parser.add_argument("--day", default=None, help="Scoring day in YYYY-MM-DD format.")
    return parser.parse_args()


def build_section(frame: pd.DataFrame, *, threshold: float, limit: int, day: str | None) -> str:
    filtered = frame[frame["score"] >= threshold].sort_values("score", ascending=False).head(limit)
    lines = [
        START_MARKER,
        "## Latest Papers Above 50%",
        "",
    ]
    if day:
        lines.append(f"Last updated for `{day}`.")
    else:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"Last updated at `{stamp}`.")
    lines.append("")

    if filtered.empty:
        lines.append(f"No papers reached `{threshold:.0%}` in the latest run.")
    else:
        for row in filtered.itertuples(index=False):
            title = " ".join(str(row.title).split())
            score = float(row.score)
            url = str(row.url)
            lines.append(f"- [{title}]({url}) ({score:.1%})")

    lines.extend(["", END_MARKER])
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
    required = {"url", "title", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Input parquet is missing required columns: {sorted(missing)}")

    readme_path = Path(args.readme)
    section = build_section(frame, threshold=args.threshold, limit=args.limit, day=args.day)
    updated = replace_managed_section(readme_path.read_text(encoding="utf-8"), section)
    readme_path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
