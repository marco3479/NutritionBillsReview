#!/usr/bin/env python3
"""
Collector: runs search passes and saves raw results (one file per pass).
- First line in each file is a JSON header: {"_meta": {...}}
- Subsequent lines are raw getBill payloads (one JSON per line)

Usage:
  python collector_legiscan.py --api_key YOUR_KEY --config config.yml --outdir data/raw

Notes:
- Config format mirrors the YAML you approved (Artifact 2). JSON also supported.
- Dedup within a pass by (bill_id, change_hash).
"""

import argparse, json, sys, time, datetime, os, re
from pathlib import Path
from urllib.parse import urlencode
from typing import Dict, Any, Iterable, List

import requests

try:
    import yaml  # optional; if not present, you can pass a .json config
    HAS_YAML = True
except Exception:
    HAS_YAML = False

BASE = "https://api.legiscan.com/"

def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    data = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if not HAS_YAML:
            print("ERROR: PyYAML not installed. Either install `pyyaml` or pass a .json config.", file=sys.stderr)
            sys.exit(1)
        return yaml.safe_load(data)
    return json.loads(data)

def states_from_config(cfg: Dict[str, Any]) -> List[str]:
    lst = cfg.get("jurisdictions", {}).get("list")
    if lst and isinstance(lst, list):
        return [s.strip().upper() for s in lst]
    # fallback ALL
    return ["AL","AK","AZ","AR","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"]

def years_from_config(cfg: Dict[str, Any]) -> List[int]:
    tf = cfg.get("timeframe", {})
    s, e = tf.get("start_year", 2020), tf.get("end_year", 2025)
    return list(range(int(s), int(e)+1))

def api_call(api_key: str, op: str, **params) -> Dict[str, Any]:
    url = f"{BASE}?{urlencode({'key': api_key, 'op': op, **params})}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        raise RuntimeError(f"LegiScan API error for {op}: {data}")
    return data

def get_search_raw(api_key: str, state: str, query: str, year: int, page: int) -> Dict[str, Any]:
    return api_call(api_key, "getSearchRaw", state=state, query=query, year=year, page=page)

def get_bill(api_key: str, bill_id: int) -> Dict[str, Any]:
    return api_call(api_key, "getBill", id=bill_id)

def now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _to_int(x, default=0):
    try:
        # handle strings like "3" and ints transparently
        return int(str(x).strip())
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--config", required=True, help="config.yaml or .json (Artifact 2)")
    ap.add_argument("--outdir", default="data/raw")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--override_states", default=None, help="comma list (e.g., CA,MA,NY)")
    ap.add_argument("--override_years", default=None, help="e.g., 2024 or 2023-2024")
    ap.add_argument("--override_bundles", default=None, help="comma list of bundle ids (e.g., school,ssb_labeling)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bundles = cfg.get("bundles", [])
    if args.override_bundles:
        wanted = set([b.strip() for b in args.override_bundles.split(",") if b.strip()])
        bundles = [b for b in bundles if b.get("id") in wanted]

    states = states_from_config(cfg)
    if args.override_states:
        states = [s.strip().upper() for s in args.override_states.split(",") if s.strip()]

    if args.override_years:
        if "-" in args.override_years:
            a,b = args.override_years.split("-", 1)
            years = list(range(int(a), int(b)+1))
        else:
            years = [int(args.override_years)]
    else:
        years = years_from_config(cfg)

    ensure_dir(args.outdir)
    file_pattern = cfg.get("execution", {}).get("file_naming", "{date}_{bundleId}_{state}_{year}.ndjson")

    for st in states:
        for yr in years:
            for bundle in bundles:
                bundle_id = bundle.get("id")
                query = bundle.get("query", "").strip()
                if not query:
                    continue

                # build output file path
                date_str = datetime.datetime.now().strftime("%Y%m%d")
                fname = file_pattern.format(date=date_str, bundleId=bundle_id, state=st, year=yr)
                fpath = Path(args.outdir) / fname

                # open file and write meta header
                with fpath.open("w", encoding="utf-8") as fout:
                    header = {
                        "_meta": {
                            "run_id": now_iso(),
                            "bundle_id": bundle_id,
                            "query": query,
                            "state": st,
                            "year": yr,
                            "api": "getSearchRaw/getBill",
                            "collector_version": "v1.0",
                            "note": "search-only output; parsing happens later"
                        }
                    }
                    fout.write(json.dumps(header, ensure_ascii=False) + "\n")

                    seen = set()  # (bill_id, change_hash)
                    page = 1
                    while True:
                        try:
                            sdata = get_search_raw(args.api_key, st, query, yr, page)
                        except Exception as e:
                            sys.stderr.write(f"[warn] getSearchRaw failed state={st} year={yr} page={page} bundle={bundle_id}: {e}\n")
                            break

                        sr = sdata.get("searchresult", {}) or {}
                        summary = sr.get("summary", {}) or {}
                        results = sr.get("results") or []
                        if not results:
                            break

                        for rec in results:
                            bill_id = rec.get("bill_id")
                            chash = rec.get("change_hash")
                            if not bill_id:
                                continue
                            key = (bill_id, chash)
                            if key in seen:
                                continue

                            # fetch detailed bill
                            time.sleep(args.sleep)
                            try:
                                bdetail = get_bill(args.api_key, bill_id)
                            except Exception as e:
                                sys.stderr.write(f"[warn] getBill failed bill_id={bill_id}: {e}\n")
                                continue

                            # write raw payload line
                            fout.write(json.dumps(bdetail, ensure_ascii=False) + "\n")
                            seen.add(key)

                        pc = _to_int(summary.get("page_current", 1), 1)
                        pt = _to_int(summary.get("page_total", 1), 1)

                        # optional: quick progress print
                        print(f"[pass] {bundle_id} {st} {yr} page {pc}/{pt} (+{len(results)} ids)")

                        if pc >= pt:
                            break
                        page += 1

                        time.sleep(args.sleep)

                print(f"[ok] wrote {fpath}  (unique records: {len(seen)})")

if __name__ == "__main__":
    main()
