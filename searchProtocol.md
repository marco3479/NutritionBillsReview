Search Protocol (v1.0)

Purpose: define how we query LegiScan, what we save, and how we keep runs reproducible—separating “collection” (raw search results) from “parsing/normalization” (your final schema).

Scope: state-level bills related to obesity prevention across domains (nutrition, school meals, SSB taxes/labeling, food access, PA/built environment, healthcare/workplace, maternal/child) for 2020–2025.

Data source: LegiScan API (getSearchRaw for discovery, getBill for detail; optional getBillText later via doc_id).

Run structure: one script (“collector”) performs repeated passes (domain bundles × state × year), writing one output file per pass; a second script (“parser”) reads those files and emits your final schema.

File strategy:

One file per pass: {date}_{bundleId}_{state}_{year}.ndjson

Each file begins with a single JSON “header record” capturing run metadata:
```json
{"_meta":{
   "run_id":"2025-11-05T21:15:00Z",
   "bundle_id":"school",
   "query":"(\"school meals\" OR \"school nutrition\" ...)",
   "state":"MA",
   "year":2023,
   "api":"getSearchRaw/getBill",
   "collector_version":"v1.0",
   "note":"search-only output; parsing happens later"}}
```

Records after the header are raw getBill payloads or a thin “bill index” (your choice). No normalization here.

Deduplication:

Within a pass: keep first occurrence per (bill_id, change_hash).

Across passes: the parser will dedupe again by bill_id, favoring newest status_date/change_hash.

Pagination:

Use getSearchRaw with paging until page_current == page_total.

Inclusion/exclusion by bill type:

Keep everything except resolutions (per your rule).

Exclude type IDs: {2,3,4,5,8,10,12,13,14,15,16} // Resolution, Concurrent Resolution, Joint Resolution, Joint Resolution Const. Amendment, Memorial, Commendation, Joint Memorial, Proclamation, Study Request, Address, Concurrent Memorial

Keep all others (e.g., 1 B Bill, 6 EO Executive Order, 7 CA Constitutional Amendment, 17 I Initiative, 18 PET Petition, 19 SB Study Bill, 20 IP Initiative Petition, 21 RB Repeal Bill, 22 RM Remonstration, 23 CB Committee Bill).

Keyword passes:

Execute one pass per “bundle” (see Artifact 2).

State × Year grid for 2020–2025 (51 jurisdictions including DC).

Logging:

Write one .jsonl log per run with raw API responses (thin), request URLs (or op/params), timestamps, and error notes.

Record the exact query string used for each pass in the file header (above).

Parser responsibilities (separate script):

Input = collection files; Output = normalized records in your final schema (you already approved).

Map bill.bill_type → type.

status from bill.status; include status_date.

Compute filed_date = earliest history[].date.

Choose text_url = latest texts[].url (fallback text_url).

Populate helpful context (session_*, last_action, sponsors, subjects as names).

Leave classification fields (pro, category_environment, topic, status_code) as null for coders/LLM.

Optional: fill outcome to "passed" or "failed" once terminal; otherwise null.

QC:

Random 5–10% sample: compare parser outputs vs. state site links (research_url) to validate status/date/title/number.

Reproducibility:

All run parameters and bundles are defined in a separate config (Artifact 2). The collector reads only from that config.

Then the analysis can proceed on the parsed CSV/NDJSON that matches your schema.