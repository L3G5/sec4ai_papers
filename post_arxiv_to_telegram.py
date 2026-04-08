from __future__ import annotations

import argparse
import html
import json
import logging
import os
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


LOG_DIR = Path("logs")
TELEGRAPH_API_URL = "https://api.telegra.ph"
TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAPH_PAGE_CHAR_BUDGET = 55_000
TELEGRAM_MESSAGE_CHAR_BUDGET = 3800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post scored arXiv papers to a Telegram channel and Telegraph.")
    parser.add_argument("input_path", help="Parquet file with url, abstract, score columns.")
    parser.add_argument("--bot-token", default=None, help="Telegram bot token. Falls back to TELEGRAM_BOT_TOKEN.")
    parser.add_argument("--chat-id", default=None, help="Telegram channel chat id or @channel username. Falls back to TELEGRAM_CHAT_ID.")
    parser.add_argument("--telegraph-access-token", default=None, help="Telegraph access token.")
    parser.add_argument("--telegraph-short-name", default="paperanalyzer", help="Telegraph account short name if creating one.")
    parser.add_argument("--telegraph-author-name", default="paper_analyzer", help="Telegraph author name.")
    parser.add_argument("--telegraph-author-url", default="", help="Telegraph author URL.")
    parser.add_argument("--telegram-threshold", type=float, default=0.5, help="Post directly to Telegram if score >= threshold.")
    parser.add_argument("--telegraph-threshold", type=float, default=0.1, help="Include in Telegraph only if score > threshold and below the Telegram threshold.")
    parser.add_argument("--title", default=None, help="Optional title prefix for Telegram and Telegraph posts.")
    parser.add_argument("--log-max-bytes", type=int, default=2_000_000)
    parser.add_argument("--log-backup-count", type=int, default=5)
    return parser.parse_args()


def setup_logger(log_max_bytes: int, log_backup_count: int) -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"post_arxiv_to_telegram_{timestamp}.log"

    logger = logging.getLogger("post_arxiv_to_telegram")
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


def http_post_json(url: str, data: dict[str, Any]) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(url, data=encoded, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}: {body}") from exc


def read_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"url", "abstract", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Input parquet is missing required columns: {sorted(missing)}")
    for optional_column in ["title", "authors"]:
        if optional_column not in frame.columns:
            frame[optional_column] = ""
    return frame.sort_values("score", ascending=False).reset_index(drop=True)


def ensure_telegraph_access_token(args: argparse.Namespace, logger: logging.Logger) -> str:
    if args.telegraph_access_token:
        return args.telegraph_access_token

    logger.info("Creating Telegraph account because no access token was provided")
    response = http_post_json(
        f"{TELEGRAPH_API_URL}/createAccount",
        {
            "short_name": args.telegraph_short_name,
            "author_name": args.telegraph_author_name,
            "author_url": args.telegraph_author_url,
        },
    )
    if not response.get("ok"):
        raise SystemExit(f"Telegraph createAccount failed: {response}")
    token = response["result"]["access_token"]
    logger.info("Created Telegraph account with short_name=%s", args.telegraph_short_name)
    return token


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).split())


def alpha_url(arxiv_url: str) -> str:
    paper_id = str(arxiv_url).rstrip("/").rsplit("/", 1)[-1]
    return f"https://www.alphaxiv.org/overview/{paper_id}"


def author_preview(authors: str, limit: int = 10) -> str:
    parts = [part.strip() for part in str(authors).split(",") if part.strip()]
    if len(parts) <= limit:
        return ", ".join(parts)
    return ", ".join(parts[:limit]) + ", ..."


def split_text_chunks(text: str, max_len: int) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        split_at = remaining.rfind(" ", 0, max_len)
        if split_at <= 0:
            split_at = max_len
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks


def telegram_post_messages(index: int, row: pd.Series) -> list[str]:
    pct = row["score"] * 100.0
    url = html.escape(str(row["url"]))
    title = html.escape(normalize_spaces(str(row["title"])) or str(row["url"]))
    authors = html.escape(author_preview(str(row["authors"])))
    abstract = html.escape(normalize_spaces(str(row["abstract"])))
    alpha = html.escape(alpha_url(str(row["url"])))
    icon = "🔴" if row["score"] > 0.85 else "🟡"
    header = (
        f"<b>{icon} {pct:.1f}%</b>\n"
        f"<a href=\"{url}\">{title}</a>\n"
        f"<b>Authors:</b> {authors}"
    )
    budget_for_abstract = max(200, TELEGRAM_MESSAGE_CHAR_BUDGET - len(header) - 2)
    abstract_chunks = split_text_chunks(abstract, budget_for_abstract)
    messages = [f"{header}\n{abstract_chunks[0]}\n<a href=\"{alpha}\">α overview</a>"]
    for chunk in abstract_chunks[1:]:
        messages.append(f"<b>{icon} {index}. continuation</b>\n{chunk}")
    return messages


def chunked_messages(header: str, lines: list[str]) -> list[str]:
    messages: list[str] = []
    current = header
    for line in lines:
        candidate = f"{current}\n\n{line}" if current else line
        if len(candidate) <= TELEGRAM_MESSAGE_CHAR_BUDGET:
            current = candidate
            continue
        messages.append(current)
        current = f"{header}\n\n{line}" if header else line
    if current:
        messages.append(current)
    return messages


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str,
    *,
    parse_mode: str | None = "HTML",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": "true",
    }
    if parse_mode is not None:
        payload["parse_mode"] = parse_mode
    return http_post_json(f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage", payload)


def telegraph_nodes_for_row(index: int, row: pd.Series) -> list[dict[str, Any]]:
    pct = row["score"] * 100.0
    url = str(row["url"])
    title = normalize_spaces(str(row["title"])) or url
    authors = author_preview(str(row["authors"]))
    abstract = normalize_spaces(str(row["abstract"]))
    alpha = alpha_url(url)
    return [
        {"tag": "p", "children": [{"tag": "strong", "children": [f"{index}. {pct:.1f}%"]}]},
        {"tag": "p", "children": [{"tag": "a", "attrs": {"href": url}, "children": [title]}]},
        {"tag": "p", "children": [{"tag": "strong", "children": ["Authors: "]}, authors]},
        {"tag": "p", "children": [abstract]},
        {"tag": "p", "children": [{"tag": "a", "attrs": {"href": alpha}, "children": ["α overview"]}]},
        {"tag": "hr"},
    ]


def create_telegraph_pages(
    *,
    access_token: str,
    title_prefix: str,
    rows: pd.DataFrame,
    author_name: str,
    author_url: str,
) -> list[str]:
    pages: list[str] = []
    chunks: list[list[dict[str, Any]]] = []
    current_nodes: list[dict[str, Any]] = []
    current_size = 0

    for index, row in enumerate(rows.itertuples(index=False), start=1):
        row_series = pd.Series(
            {
                "url": row.url,
                "title": getattr(row, "title", ""),
                "authors": getattr(row, "authors", ""),
                "abstract": row.abstract,
                "score": row.score,
            }
        )
        nodes = telegraph_nodes_for_row(index, row_series)
        node_size = len(json.dumps(nodes, ensure_ascii=False))
        if current_nodes and current_size + node_size > TELEGRAPH_PAGE_CHAR_BUDGET:
            chunks.append(current_nodes)
            current_nodes = []
            current_size = 0
        current_nodes.extend(nodes)
        current_size += node_size

    if current_nodes:
        chunks.append(current_nodes)

    for page_index, content_nodes in enumerate(chunks, start=1):
        page_title = title_prefix if len(chunks) == 1 else f"{title_prefix} ({page_index}/{len(chunks)})"
        response = http_post_json(
            f"{TELEGRAPH_API_URL}/createPage",
            {
                "access_token": access_token,
                "title": page_title,
                "author_name": author_name,
                "author_url": author_url,
                "content": json.dumps(content_nodes, ensure_ascii=False),
                "return_content": "false",
            },
        )
        if not response.get("ok"):
            raise SystemExit(f"Telegraph createPage failed: {response}")
        pages.append(response["result"]["url"])
    return pages


def main() -> None:
    load_dotenv()
    args = parse_args()
    bot_token = args.bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = args.chat_id or os.getenv("TELEGRAM_CHAT_ID")
    telegraph_access_token = args.telegraph_access_token or os.getenv("TELEGRAPH_ACCESS_TOKEN")
    telegraph_author_name = args.telegraph_author_name or os.getenv("TELEGRAPH_AUTHOR_NAME", "paper_analyzer")
    telegraph_author_url = args.telegraph_author_url or os.getenv("TELEGRAPH_AUTHOR_URL", "")
    telegraph_short_name = args.telegraph_short_name or os.getenv("TELEGRAPH_SHORT_NAME", "paperanalyzer")
    if not bot_token:
        raise SystemExit("Missing Telegram bot token. Set --bot-token or TELEGRAM_BOT_TOKEN in .env.")
    if not chat_id:
        raise SystemExit("Missing Telegram chat id. Set --chat-id or TELEGRAM_CHAT_ID in .env.")
    if args.telegraph_threshold >= args.telegram_threshold:
        raise SystemExit("--telegraph-threshold must be below --telegram-threshold.")

    logger, log_path = setup_logger(args.log_max_bytes, args.log_backup_count)
    logger.info("Starting Telegram publish")
    logger.info(
        "Input=%s TelegramThreshold=%.3f TelegraphThreshold=%.3f Chat=%s",
        args.input_path,
        args.telegram_threshold,
        args.telegraph_threshold,
        chat_id,
    )

    frame = read_scores(Path(args.input_path))
    high = frame[frame["score"] >= args.telegram_threshold].copy()
    medium = frame[(frame["score"] > args.telegraph_threshold) & (frame["score"] < args.telegram_threshold)].copy()
    dropped = frame[frame["score"] <= args.telegraph_threshold].copy()
    title_prefix = args.title or f"arXiv papers {datetime.now().strftime('%Y-%m-%d')}"

    logger.info(
        "Loaded %s rows: %s Telegram, %s Telegraph, %s dropped",
        len(frame),
        len(high),
        len(medium),
        len(dropped),
    )

    messages: list[str] = []
    intro = f"<b>{html.escape(title_prefix)}</b>"
    messages.append(intro)
    if not high.empty:
        for index, (_, row) in enumerate(high.iterrows(), start=1):
            messages.extend(telegram_post_messages(index, row))
        logger.info("Sending %s Telegram message(s) for high-score papers", len(messages))
    else:
        logger.info("No papers above threshold; skipping direct Telegram paper posts")

    for idx, message in enumerate(messages, start=1):
        response = send_telegram_message(bot_token, chat_id, message)
        if not response.get("ok"):
            raise SystemExit(f"Telegram sendMessage failed: {response}")
        logger.info("Sent Telegram message %s/%s", idx, len(messages))

    telegraph_urls: list[str] = []
    if not medium.empty:
        args.telegraph_access_token = telegraph_access_token
        args.telegraph_author_name = telegraph_author_name
        args.telegraph_author_url = telegraph_author_url
        args.telegraph_short_name = telegraph_short_name
        token = ensure_telegraph_access_token(args, logger)
        telegraph_urls = create_telegraph_pages(
            access_token=token,
            title_prefix=f"{title_prefix} Telegraph archive",
            rows=medium,
            author_name=telegraph_author_name,
            author_url=telegraph_author_url,
        )
        summary_lines = [
            f"Telegraph ({len(medium)} paper(s), {args.telegraph_threshold:.0%} < score < {args.telegram_threshold:.0%}):"
        ] + telegraph_urls
        if not dropped.empty:
            summary_lines.append(f"Dropped {len(dropped)} paper(s) with score <= {args.telegraph_threshold:.0%}.")
        summary_message = "\n".join(summary_lines)
        response = send_telegram_message(bot_token, chat_id, summary_message, parse_mode=None)
        if not response.get("ok"):
            raise SystemExit(f"Telegram summary sendMessage failed: {response}")
        logger.info("Sent Telegraph summary message with %s page(s)", len(telegraph_urls))
    else:
        logger.info("No papers in the Telegraph score band; skipping Telegraph")
        if not dropped.empty:
            dropped_message = f"Dropped {len(dropped)} paper(s) with score <= {args.telegraph_threshold:.0%}."
            response = send_telegram_message(bot_token, chat_id, dropped_message, parse_mode=None)
            if not response.get("ok"):
                raise SystemExit(f"Telegram dropped-summary sendMessage failed: {response}")

    print(f"Posted {len(high)} high-score papers directly to Telegram")
    print(f"Created {len(telegraph_urls)} Telegraph page(s) for lower-score papers")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
