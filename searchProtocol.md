# Search Protocol (v1.0)

## Purpose
- Define how we query LegiScan, what we save, and how we keep runs reproducible.  
- Keep “collection” (raw search results) separate from “parsing/normalization” (final schema).

## Scope
- Focus on state-level bills touching obesity prevention domains: nutrition, school meals, SSB taxes/labeling, food access, PA/built environment, healthcare/workplace, maternal/child.  
- Time horizon: 2020–2025.

## Data Source
- LegiScan API.  
  - `getSearchRaw` for discovery.  
  - `getBill` for detailed payloads.  
  - `getBillText` optional later via `doc_id`.

## Run Structure
1. **Collector (`collector_legiscan.py`)** – executes one pass per bundle and state/year (or the national index when `--state_scope all` is set), writing a single NDJSON file per pass.  
2. **Parser (`parser_legiscan_to_schema.py`)** – reads those NDJSON files, dedupes across passes, and emits your normalized schema.  
3. **Keyword pre-screen (`filterRelatedToNutrition.py`)** – annotates summaries with a fast binary `related` flag via keyword matching.  
4. **Classifier (`classify.py`)** – batches normalized rows to the OpenAI Responses API and writes the nutrition-policy labels back into the workbook.

## File Strategy
- File naming: `{date}_{bundleId}_{state}_{year}.ndjson` (one per pass).  
- First line is a JSON header with run metadata; subsequent lines are raw `getBill` payloads (one JSON object per line).

### Header Example
```json
{"_meta":{
  "run_id":"2025-11-05T21:15:00Z",
  "bundle_id":"school",
  "query":"(\"school meals\" OR \"school nutrition\" ...)",
  "state":"MA",
  "year":2023,
  "api":"getSearchRaw/getBill",
  "collector_version":"v1.0",
  "cache_reused":22,
  "note":"search-only output; parsing happens later"}}
```

## Deduplication
- **Within a pass** – keep only the first occurrence per `(bill_id, change_hash)` (handled by the collector).  
- **Across passes** – parser dedupes again by `bill_id`, preferring newer `status_date`, then newer `change_hash` when dates tie.

## Pagination & Fetching
- Page through `getSearchRaw` until `page_current == page_total`.  
- Default mode uses `--state_scope all`, hitting the national index once per bundle/year; use `--state_scope per-state` (or override states) to run jurisdiction-by-jurisdiction.  
- Bill detail fetches use optional async batching (`aiohttp`) with throttled concurrency; if `aiohttp` is unavailable the collector falls back to sequential requests.  
- Live progress messages report both page traversal `[pass ...]` and bill detail downloads `[fetch ...]`.
- When a `(bill_id, change_hash)` combination already exists in any NDJSON within the target outdir, that payload is reused automatically—no additional `getBill` call is made (even across bundles).

## Inclusion / Exclusion
- Keep everything except resolution-class bill types.  
- Exclude type IDs: `{2, 3, 4, 5, 8, 10, 12, 13, 14, 15, 16}`  → Resolution, Concurrent Resolution, Joint Resolution, Joint Resolution Const. Amendment, Memorial, Commendation, Joint Memorial, Proclamation, Study Request, Address, Concurrent Memorial.  
- Keep all others (e.g., `1 B Bill`, `6 EO Executive Order`, `7 CA Constitutional Amendment`, `17 I Initiative`, `18 PET Petition`, `19 SB Study Bill`, `20 IP Initiative Petition`, `21 RB Repeal Bill`, `22 RM Remonstration`, `23 CB Committee Bill`).

## Classification Pass
- Source: parser output workbook (typically `data/parsed/NutritionBillsReview.xlsx`) plus the keyword-derived `related` column inserted by `filterRelatedToNutrition.py`.  
- Transport: OpenAI `/v1/responses` batches with JSON-schema enforcement.  
- Defaults: dynamic request sizing capped by `--max-batch-kb` (20 KB) and `--max-output-tokens` (5000).  
- Required env: `OPENAI_API_KEY` or explicit `--api-key`.  
- Reruns: use `--resume-batch-id <batch_id>` to skip resubmission and rehydrate an existing batch’s output.  
- Debugging: set `--retain-request-file` (keeps the JSONL sent to OpenAI) and `--verbose` for HTTP tracing; batch errors are logged from the returned `error_file_ids`.

## Keyword Passes
- Execute one pass per bundle (see Artifact 2 for bundle definitions).  
- Cover the full state × year grid for 2020–2025 (51 jurisdictions including DC) unless overrides are supplied.

## Logging Expectations
- Emit a `.jsonl` log per run with request parameters, timestamps, and error notes.  
- The collector header already records the exact query string for each pass (ensure those are preserved for reproducibility).

## Parser Responsibilities
- Input: collector NDJSON files.  
- Output: normalized NDJSON + CSV (matching the approved schema).  
- Key field mappings:
  - Map `bill.bill_type` → human-readable type.  
  - Copy `bill.status` / `status_date`.  
  - `filed_date` = earliest `history[].date`.  
  - `text_url` = newest `texts[].url` (fallback `bill.text_url`).  
  - Populate contextual fields (`session_*`, `last_action`, `sponsors`, `subjects`, etc.).  
  - Leave coder fields (`pro`, `category_environment`, `topic`, `status_code`) as `null`.  
  - Set `outcome` to `"passed"` or `"failed"` once terminal; otherwise leave `null`.

## Quality Control
- Manually review a random 5–10 % sample against authoritative state links (`research_url`) to verify number, title, status, and dates.
- Spot-check classifier results by comparing `related_to_nutrition` and topic assignments against source text for at least one batch per run.

## Reproducibility
- All bundles, timeframe, and inclusion rules live in config files (Artifact 2).  
- The collector reads only from that config, ensuring runs can be reproduced later.  
- Downstream analysis should only operate on the normalized NDJSON/CSV emitted by the parser.