#!/usr/bin/env python3
"""
Parser: reads collector NDJSON files, dedupes across passes, applies bill-type policy,
and emits your final schema as NDJSON + CSV.

Usage:
  python parser_legiscan_to_schema.py \
      --inputs data/raw/*.ndjson \
      --config config.yml \
      --out_ndjson data/parsed/bills.ndjson \
      --out_csv data/parsed/bills.csv
"""

import argparse, csv, glob, json, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False

STATUS_MAP: Dict[int, str] = {
    0: "pre-filed",
    1: "introduced",
    2: "engrossed",
    3: "enrolled",
    4: "passed",
    5: "vetoed",
    6: "failed",
    7: "override",
    8: "chaptered",
    9: "refer",
    10: "report-pass",
    11: "report-dnp",
    12: "draft"
}

# Bill type ids (from your list)
BILL_TYPE_LABELS: Dict[str, str] = {
    "B": "Bill",
    "R": "Resolution",
    "CR": "Concurrent Resolution",
    "JR": "Joint Resolution",
    "JRCA": "Joint Resolution Constitutional Amendment",
    "EO": "Executive Order",
    "CA": "Constitutional Amendment",
    "M": "Memorial",
    "CL": "Claim",
    "C": "Commendation",
    "CSR": "Committee Study Request",
    "JM": "Joint Memorial",
    "P": "Proclamation",
    "SR": "Study Request",
    "A": "Address",
    "CM": "Concurrent Memorial",
    "I": "Initiative",
    "PET": "Petition",
    "SB": "Study Bill",
    "IP": "Initiative Petition",
    "RB": "Repeal Bill",
    "RM": "Remonstration",
    "CB": "Committee Bill",
}

def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    data = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if not HAS_YAML:
            print("ERROR: PyYAML not installed. Either install `pyyaml` or use a .json config.", file=sys.stderr)
            sys.exit(1)
        return yaml.safe_load(data)
    return json.loads(data)

def newest_text_url(bill: Dict[str, Any]) -> str:
    texts = bill.get("texts") or []
    if texts:
        # pick the newest by `date` if present
        texts = sorted(texts, key=lambda t: (t.get("date") or ""), reverse=True)
        for t in texts:
            url = t.get("url")
            if url: return url
    return bill.get("text_url")

def earliest_history_date(bill: Dict[str, Any]) -> str:
    hist = bill.get("history") or []
    dates = [h.get("date") for h in hist if h.get("date")]
    return min(dates) if dates else None

def latest_history_action(bill: Dict[str, Any]) -> Tuple[str, str]:
    hist = bill.get("history") or []
    if not hist:
        return (None, None)
    h = hist[-1]
    return (h.get("action"), h.get("date"))

def to_schema(bill_payload: Dict[str, Any]) -> Dict[str, Any]:
    b = bill_payload.get("bill", {}) or {}
    sess = b.get("session") or {}
    sponsors = b.get("sponsors") or []
    subjects = b.get("subjects") or []

    bill_type_id: str = b.get("bill_type")
    bill_type_label: str = BILL_TYPE_LABELS.get(bill_type_id, None)

    status_txt: str = STATUS_MAP.get(b.get("status"), "pending")
    status_date = b.get("status_date")

    # outcome only when terminal
    if status_txt == "passed":
        outcome = "passed"
    elif status_txt in ("failed", "vetoed"):
        outcome = "failed"
    else:
        outcome = None

    last_action, last_action_date = latest_history_action(b)

    schema = {
        # BILL INFO
        "id": str(b.get("bill_id")) if b.get("bill_id") is not None else None,
        "number": b.get("number") or b.get("bill_number"),
        "state": b.get("state"),
        "text_url": newest_text_url(b),
        "summary": b.get("description") or b.get("title"),
        "type": bill_type_label,
        "document_url": b.get("url"),
        "filed_date": earliest_history_date(b),

        "author": sponsors[0].get("name") if sponsors else None,
        "sponsor_party": sponsors[0].get("party") if sponsors else None,
        "sponsors": [s.get("name") for s in sponsors if s.get("name")],
        "subjects": [s.get("subject_name") or s.get("subject") for s in subjects if (s.get("subject_name") or s.get("subject"))],

        "cycle": int(f"{sess.get('year_start')}{sess.get('year_end')}") if (sess.get("year_start") and sess.get("year_end")) else None,
        "status": status_txt,
        "status_date": status_date,

        "session_name": sess.get("session_name"),
        "session_year_start": sess.get("year_start"),
        "session_year_end": sess.get("year_end"),
        "last_action": last_action,
        "last_action_date": last_action_date,
        "change_hash": b.get("change_hash"),
        "text_doc_ids": [t.get("doc_id") for t in (b.get("texts") or []) if t.get("doc_id") is not None],

        # POLICY CLASSIFICATION (manual later)
        "pro": None,
        "outcome": outcome,
        "status_code": None,
        "category_environment": None,
        "topic": None
    }
    return schema

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="glob(s) like data/raw/*.ndjson")
    ap.add_argument("--config", required=True, help="same config as collector (YAML/JSON)")
    ap.add_argument("--out_ndjson", required=True)
    ap.add_argument("--exclusion_report_json", default=None, help="Optional path to write exclusion-by-type JSON")
    ap.add_argument("--exclusion_report_csv", default=None, help="Optional path to write exclusion-by-type CSV")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # resolution-class exclusions
    exclude_types = set(cfg.get("bill_type_policy", {}).get("exclude_type_ids", []))


    from collections import defaultdict

    excluded_by_type = defaultdict(int)   # key: bill_type_id
    kept_by_type = defaultdict(int)       # optional, for visibility
    total_seen = 0
    total_excluded = 0

    files: List[Path] = []
    for pattern in args.inputs:
        files.extend([Path(p) for p in glob.glob(pattern)])
    files = sorted(files)

    # cross-pass dedupe key: prefer newer status_date, then newer change_hash if tie
    best: Dict[str, Dict[str, Any]] = {}      # bill_id -> schema row
    best_meta: Dict[str, Dict[str, Any]] = {} # bill_id -> {status_date, change_hash}

    for fp in files:
        with fp.open("r", encoding="utf-8") as fin:
            first = True
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                # skip header
                if first and "_meta" in data:
                    first = False
                    continue
                first = False

                # must be raw getBill payload
                bill = data.get("bill") or {}
                bill_id = bill.get("bill_id")
                if bill_id is None:
                    continue

                # Filter: keep everything except resolution-class types
                bt = bill.get("bill_type")
                total_seen += 1
                if bt in exclude_types:
                    excluded_by_type[bt] += 1
                    total_excluded += 1
                    continue
                else:
                    kept_by_type[bt] += 1


                row = to_schema(data)

                # Dedup logic across passes
                row_status_date = row.get("status_date") or ""
                row_change_hash = row.get("change_hash") or ""

                prev = best_meta.get(str(bill_id))
                if not prev:
                    best[str(bill_id)] = row
                    best_meta[str(bill_id)] = {"status_date": row_status_date, "change_hash": row_change_hash}
                else:
                    # prefer newer status_date; if tie, prefer newer change_hash lexically
                    if row_status_date > prev["status_date"]:
                        best[str(bill_id)] = row
                        best_meta[str(bill_id)] = {"status_date": row_status_date, "change_hash": row_change_hash}
                    elif row_status_date == prev["status_date"] and row_change_hash > prev["change_hash"]:
                        best[str(bill_id)] = row
                        best_meta[str(bill_id)] = {"status_date": row_status_date, "change_hash": row_change_hash}

    # Write outputs
    Path(args.out_ndjson).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # NDJSON
    with Path(args.out_ndjson).open("w", encoding="utf-8") as fnd:
        for _, row in sorted(best.items()):
            fnd.write(json.dumps(row, ensure_ascii=False) + "\n")

    # CSV
    field_order = [
        # BILL INFO
        "id","number","state","text_url","summary","type","document_url",
        "filed_date","author","sponsor_party","sponsors","subjects","cycle","status","status_date",
        "session_name","session_year_start","session_year_end","last_action","last_action_date",
        "change_hash","text_doc_ids",
        # POLICY CLASSIFICATION
        "pro","outcome","status_code","category_environment","topic"
    ]

    def _as_str(v):
        if isinstance(v, (list, tuple)):
            return "; ".join([str(x) for x in v])
        return v

    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=field_order)
        w.writeheader()
        for _, row in sorted(best.items()):
            out = {k: _as_str(row.get(k)) for k in field_order}
            w.writerow(out)

    print(f"[ok] wrote {args.out_ndjson} and {args.out_csv} (records: {len(best)})")



    def _label(bt_id):
        return BILL_TYPE_LABELS.get(bt_id, f"Unknown({bt_id})")

    print("\n[summary] bill-type exclusion results")
    print(f"  total seen (across passes): {total_seen}")
    print(f"  total excluded by type:     {total_excluded}")

    if excluded_by_type:
        print("  excluded by type:")
        for bt_id in sorted(excluded_by_type.keys()):
            print(f"    - {bt_id:>2} {_label(bt_id):<40} : {excluded_by_type[bt_id]}")

    # Optional: also show what was kept, for context
    kept_total = sum(kept_by_type.values())
    print(f"  total kept after type filter: {kept_total}")
    if kept_by_type:
        print("  kept by type:")
        for bt_id in sorted(kept_by_type.keys()):
            print(f"    - {bt_id:>2} {_label(bt_id):<40} : {kept_by_type[bt_id]}")


    # Build a serializable report dict
    excl_report = {
        "total_seen": total_seen,
        "total_excluded": total_excluded,
        "excluded_by_type": [
            {"bill_type_id": bt, "bill_type_label": _label(bt), "count": cnt}
            for bt, cnt in sorted(excluded_by_type.items())
        ],
        "kept_by_type": [
            {"bill_type_id": bt, "bill_type_label": _label(bt), "count": cnt}
            for bt, cnt in sorted(kept_by_type.items())
        ],
    }

    if args.exclusion_report_json:
        Path(args.exclusion_report_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.exclusion_report_json).open("w", encoding="utf-8") as fjson:
            json.dump(excl_report, fjson, ensure_ascii=False, indent=2)
        print(f"[ok] wrote exclusion report JSON -> {args.exclusion_report_json}")

    if args.exclusion_report_csv:
        import csv as _csv
        Path(args.exclusion_report_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.exclusion_report_csv).open("w", newline="", encoding="utf-8") as fcsv:
            w = _csv.DictWriter(fcsv, fieldnames=["bill_type_id","bill_type_label","count","bucket"])
            w.writeheader()
            for bt, cnt in sorted(excluded_by_type.items()):
                w.writerow({"bill_type_id": bt, "bill_type_label": _label(bt), "count": cnt, "bucket": "excluded"})
            for bt, cnt in sorted(kept_by_type.items()):
                w.writerow({"bill_type_id": bt, "bill_type_label": _label(bt), "count": cnt, "bucket": "kept"})
        print(f"[ok] wrote exclusion report CSV  -> {args.exclusion_report_csv}")


if __name__ == "__main__":
    main()
