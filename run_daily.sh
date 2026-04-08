#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load local dotenv values for shell gating when present.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

DAY="${1:-$(date -u +%F)}"
OUT="artifacts/arxiv_${DAY}.parquet"

uv run python score_arxiv_api_day.py --day "$DAY" --output "$OUT"

if [[ "${UPDATE_README:-}" == "1" ]]; then
  uv run python update_readme.py "$OUT" --day "$DAY"
fi

if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
  uv run python post_arxiv_to_telegram.py "$OUT"
else
  echo "Skipping Telegram/Telegraph publish because TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set."
fi
