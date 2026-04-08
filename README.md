# ai_sec_papers_public

Public daily runner for scoring fresh arXiv papers and optionally posting the results to Telegram and Telegraph.

<!-- README_LATEST_PAPERS_START -->
## Latest Papers Above 50%

Last updated for `2026-04-08`.

No papers reached `50%` in the latest run.

<!-- README_LATEST_PAPERS_END -->

## What is included

- `score_arxiv_api_day.py`: fetches recent arXiv IDs, hydrates them via the export API, and scores them with the bundled model
- `post_arxiv_to_telegram.py`: posts high-score papers to Telegram and medium-score papers to Telegraph
- `update_readme.py`: refreshes the public README section from the latest scored parquet
- `arxiv_priority_predictor.py`: predictor wrapper used by the scoring job
- `artifacts/arxiv_priority_predictor_2511_abstract_authors_surnames.joblib`: trained model bundle
- `.github/workflows/daily.yml`: scheduled GitHub Actions workflow
- `run_daily.sh`: local or CI entrypoint for one daily run

## GitHub Actions schedule

The workflow is configured with:

```cron
0 16 * * *
```

That runs once a day at `16:00 UTC` and also supports manual runs through `workflow_dispatch`.

## Repository secrets

Set these in GitHub repository settings if you want publishing enabled:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TELEGRAPH_ACCESS_TOKEN` (optional; if omitted the script creates a Telegraph account at runtime)
- `TELEGRAPH_AUTHOR_NAME` (optional)
- `TELEGRAPH_AUTHOR_URL` (optional)
- `TELEGRAPH_SHORT_NAME` (optional)

If the Telegram secrets are missing, the workflow still runs scoring and uploads the parquet/log artifacts, but it skips publishing.

## Local run

```bash
uv sync
./run_daily.sh
```

To run a specific UTC day:

```bash
./run_daily.sh 2026-04-09
```
