# Nutrition Bills Review – Data Pipeline

## Overview

This repository automates the collection, normalization, and policy classification of U.S. nutrition and obesity‑related legislation. It has three stages:

1. **Collector (`collector_legiscan.py`)** – queries the LegiScan API, walks through result pages, and saves the raw `getBill` payloads (one JSON per line) for later processing.  
2. **Parser (`parser_legiscan_to_schema.py`)** – turns the raw payloads into your working schema (NDJSON + CSV), deduping across passes and applying your bill‑type rules.  
3. **Keyword pre-screen (`filterRelatedToNutrition.py`)** – quickly annotates each summary with a binary `related` flag via keyword/regex matching.  
4. **Classifier (`classify.py`)** – calls the OpenAI Responses API to label each bill with nutrition relevance, directionality, environment, and topic.

The workflow is intentionally lightweight: configure once, run the collector whenever you want fresh data, then re-run the parser to regenerate normalized outputs.

---

## Requirements

- Python 3.9+  
- `requests` (collector)  
- `PyYAML` (optional; required when configs are YAML instead of JSON)  
- `aiohttp` (optional; enables concurrent `getBill` requests – otherwise the collector gracefully falls back to sequential fetching)  
- OpenAI Python SDK (`pip install openai`) for the classifier stage

Install dependencies with:

```bash
pip install -r requirements.txt  # or pip install requests pyyaml aiohttp
```

---

## Configuration Files

Two configs live in the repo:

- `config.yml` – production run (all jurisdictions, all bundles, full timeframe).  
- `config.test.yml` – smoke test (Massachusetts 2024, condensed query).

Key blocks shared by both:

- `timeframe.start_year` / `end_year`
- `jurisdictions.list` – default set of state codes. The collector keeps this list unless you pass `--state_scope all` or `--override_states`.
- `bundles` – query definitions. Each bundle is run for every state/year combination unless overridden.
- `execution` – documentation for expected paging size, file naming, etc.
- `parser_output` – schema expectations used by the parser.

---

## Collector Usage

```
python collector_legiscan.py \
    --api_key YOUR_KEY \
    --config config.yml \
    --outdir data/raw \
    --sleep 0.4 \
    --state_scope all \
    --async_workers 4 \
    --progress_interval 25
```

### Important Flags

| Flag | Description |
| --- | --- |
| `--api_key` | LegiScan API key. Required. |
| `--config` | Path to YAML/JSON config (test/production). |
| `--outdir` | Destination directory for ndjson artifacts. |
| `--sleep` | Delay between API calls (seconds). Used by async throttling too. |
| `--async_workers` | Max concurrent `getBill` requests when `aiohttp` is available; default `4`. Set to `1` to disable concurrency. |
| `--progress_interval` | Print fetch progress every N new bill payloads (default `50`, set `<=0` to disable). |
| `--state_scope` | `all` (default) hits LegiScan’s national index once per bundle/year. Use `per-state` to iterate `jurisdictions.list`. |
| `--override_states`, `--override_years`, `--override_bundles` | Narrow the run to specific slices. |

### Output Structure

Each collector run emits one NDJSON file per `(bundle, state, year)`:

```
{date}_{bundleId}_{state}_{year}.ndjson
```

First line is a `_meta` header with run metadata. Subsequent lines are raw `getBill` payloads (as returned by LegiScan). These are what the parser consumes.

### Performance & Caching

- **Persistent cache** – reruns reuse existing `(bill_id, change_hash)` records from the prior NDJSON files (across bundles in the same outdir), skipping redundant `getBill` calls whenever a payload already exists.  
- **Async detail fetching** – when `aiohttp` is installed, `fetch_bills_async()` pipelines individual bill downloads with a configurable concurrency limit and throttled delay.  
- **Progress logging** – you get `[fetch] … fetched=X/Y …` messages as details arrive and `[pass] … page A/B` after each `getSearchRaw` pass, even when everything is cached.  
- **Quota handling** – API errors (e.g., quota exceeded) are logged and the current bundle/year loop exits. Cached data remains intact.

---

## Parser Usage

```
python parser_legiscan_to_schema.py \
    --inputs "data/raw/*.ndjson" \
    --config config.yml \
    --out_ndjson data/parsed/bills.ndjson \
    --out_csv data/parsed/bills.csv \
    --exclusion_report_json data/parsed/exclusion_report.json \
    --exclusion_report_csv data/parsed/exclusion_report.csv
```

### Parser Responsibilities

- Dedupes across multiple collector passes, favoring the latest status date / change hash for each `bill_id`.
- Applies bill-type exclusion rules from `config.*.yml` (e.g., skipping resolutions) while counting what was kept vs excluded.
- Normalizes payloads into the fields defined under `parser_output.fields`.
- Produces:
  - NDJSON of normalized rows (`out_ndjson`)
  - CSV with flattened values (`out_csv`)
  - Optional exclusion reports (JSON + CSV) summarizing totals by bill type.

---

## Classifier Usage

```
python classify.py \
    data/parsed/NutritionBillsReview.xlsx \
    --sheet Edit3 \
    --model gpt-5.1 \
    --max-batch-kb 40 \
    --max-output-tokens 5000 \
    --retain-request-file
```

### What the classifier does

- Sends records to the OpenAI `/v1/responses` batch API using JSON-schema enforced outputs.  
- Adds four columns to the target sheet: `related_to_nutrition`, `directionality`, `environment`, `topic`.  
- Dynamically sizes each batch request by byte size unless `--batch-size` is supplied.  
- Logs progress while polling batches and surfaces API error payloads for debugging.

### Key Flags

| Flag | Description |
| --- | --- |
| `--model` | OpenAI model to invoke (default `gpt-5.1`). |
| `--max-batch-kb` | Upper bound (in KB) for each request when dynamically batching (default `20`). Lower values reduce truncation risk. |
| `--batch-size` | Fixed number of records per request. Overrides dynamic sizing when supplied. |
| `--max-output-tokens` | Response token allowance (default `5000`). Increase for longer batches, decrease to limit cost. |
| `--retain-request-file` | Keep the generated JSONL sent to OpenAI (useful when debugging). |
| `--resume-batch-id` | Skip submission and hydrate results from an existing batch id. |

### Workflow Tips

- Start with a dry run: `--limit 25 --retain-request-file --verbose`.  
- If you see `json.JSONDecodeError` about an unterminated string, reduce `--max-batch-kb` or raise `--max-output-tokens`.  
- Provide an `OPENAI_API_KEY` environment variable or pass `--api-key`.  
- The script overwrites the input workbook unless `--output` is supplied.

---

## Recommended Workflow

1. **Smoke test** (optional but quick):
   ```bash
   python collector_legiscan.py --api_key ... --config config.test.yml --outdir data/raw_test --sleep 0.4 --state_scope per-state --progress_interval 5
   python parser_legiscan_to_schema.py --inputs "data/raw_test/*.ndjson" --config config.test.yml --out_ndjson data/parsed_test/bills.ndjson --out_csv data/parsed_test/bills.csv
   ```
2. **Production run**:
   ```bash
   python collector_legiscan.py --api_key ... --config config.yml --outdir data/raw --sleep 0.4 --state_scope all
   python parser_legiscan_to_schema.py --inputs "data/raw/*.ndjson" --config config.yml --out_ndjson data/parsed/bills.ndjson --out_csv data/parsed/bills.csv
   ```
3. **Keyword pass** (optional but recommended before LLM classification):
   ```bash
   python filterRelatedToNutrition.py data/parsed/NutritionBillsReview.xlsx --sheet Edit3
   ```
   Adds/overwrites a `related` column based on summary keyword hits, giving you a quick sanity check and a baseline flag.  
4. **LLM classification**:
   ```bash
   python classify.py data/parsed/NutritionBillsReview.xlsx --sheet Edit3 --max-batch-kb 40 --max-output-tokens 5000
   ```
5. **Reruns** are safe – cached records dedupe automatically. Delete or move the NDJSON output if you want a full refetch.

---

## Troubleshooting & Tips

- **Quota exceeded** – LegiScan returns an error payload when the 30 k monthly limit is exhausted. The collector logs a warning and moves on; resume the run after the quota resets.
- **Async fallback** – If `aiohttp` is missing or an async batch raises, the collector warns and switches to sequential `requests` calls automatically.
- **Paging slowness** – When everything is cached, you’ll still see `[pass ... page A/B]` messages while `getSearchRaw` walks pages. Use a smaller `--sleep` or limit scope to speed through those checks.
- **New features** – If you want incremental caching, resume-capable pagination, or smarter backoff when the API throttles, search for `TODO:` tags and comments in the collector to see potential next steps.

---

## Project Structure

```
collector_legiscan.py        # data ingestion from LegiScan API
parser_legiscan_to_schema.py # normalization & schema output
config.yml                   # production configuration
config.test.yml              # smoke-test configuration
commands.md                  # quick command references
data/                        # raw and parsed outputs (gitignored by default)
docs/                        # supplemental research artifacts
```

Feel free to open issues or drop TODO comments inline if you extend the pipeline. Happy collecting!

