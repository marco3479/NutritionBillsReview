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

import argparse, json, sys, time, datetime, os, re, asyncio
from pathlib import Path
from urllib.parse import urlencode
from typing import Dict, Any, Iterable, List, Set, Tuple, Optional

import requests

try:
    import yaml  # optional; if not present, you can pass a .json config
    HAS_YAML = True
except Exception:
    HAS_YAML = False

try:
    import aiohttp  # type: ignore  # optional; enables async bill fetching
    HAS_AIOHTTP = True
except Exception:
    aiohttp = None
    HAS_AIOHTTP = False

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


class AsyncThrottle:
    def __init__(self, interval: float):
        self.interval = max(interval or 0.0, 0.0)
        self._next_ts = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            if now < self._next_ts:
                await asyncio.sleep(self._next_ts - now)
                now = time.monotonic()
            self._next_ts = now + self.interval


async def fetch_bills_async(
    api_key: str,
    bill_ids: Iterable[int],
    interval: float,
    max_concurrency: int,
    progress_hook=None,
) -> Dict[int, Any]:
    timeout = aiohttp.ClientTimeout(total=60)
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    throttle = AsyncThrottle(interval)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def worker(bid: int) -> Dict[str, Any]:
            async with semaphore:
                await throttle.wait()
                params = {"key": api_key, "op": "getBill", "id": bid}
                async with session.get(BASE, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    if data.get("status") != "OK":
                        raise RuntimeError(f"LegiScan API error for getBill: {data}")
                    if progress_hook:
                        progress_hook(bid)
                    return data

        tasks = {bid: asyncio.create_task(worker(bid)) for bid in bill_ids}
        results: Dict[int, Any] = {}
        for bid, task in tasks.items():
            try:
                results[bid] = await task
            except Exception as exc:
                results[bid] = exc
                if progress_hook:
                    progress_hook(bid, error=exc)
        return results


def load_existing_cache(path: Path) -> Tuple[Set[Tuple[int, str]], List[str]]:
    seen: Set[Tuple[int, str]] = set()
    payload_lines: List[str] = []
    if not path.exists():
        return seen, payload_lines

    try:
        with path.open("r", encoding="utf-8") as fin:
            for idx, raw in enumerate(fin):
                raw = raw.rstrip("\n")
                if idx == 0:
                    continue  # skip header
                if not raw:
                    continue
                payload_lines.append(raw)
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                bill = obj.get("bill") or {}
                bill_id = bill.get("bill_id")
                change_hash = bill.get("change_hash")
                if bill_id and change_hash:
                    seen.add((bill_id, change_hash))
    except Exception as exc:
        print(f"[warn] failed to read existing cache {path}: {exc}", file=sys.stderr)
        return set(), []

    return seen, payload_lines


def build_directory_cache(outdir: Path) -> Dict[Tuple[int, str], str]:
    cache: Dict[Tuple[int, str], str] = {}
    if not outdir.exists():
        return cache

    for path in sorted(outdir.glob("*.ndjson")):
        try:
            with path.open("r", encoding="utf-8") as fin:
                first = True
                for line in fin:
                    line = line.rstrip("\n")
                    if first:
                        first = False
                        continue
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    bill = obj.get("bill") or {}
                    bill_id = bill.get("bill_id")
                    change_hash = bill.get("change_hash")
                    if bill_id and change_hash:
                        key = (bill_id, change_hash)
                        if key not in cache:
                            cache[key] = line
        except Exception as exc:
            print(f"[warn] failed to load cache from {path}: {exc}", file=sys.stderr)
            continue

    return cache

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
    ap.add_argument(
        "--async_workers",
        type=int,
        default=4,
        help="Maximum concurrent getBill requests when aiohttp is installed (set to 1 to disable).",
    )
    ap.add_argument(
        "--progress_interval",
        type=int,
        default=50,
        help="Print a progress message every N new bill payloads fetched (<=0 disables).",
    )
    ap.add_argument(
        "--state_scope",
        choices=["all", "per-state"],
        default="all",
        help="Use 'all' to search the national index once (default) or 'per-state' to search each jurisdiction individually.",
    )
    ap.add_argument("--override_states", default=None, help="comma list (e.g., CA,MA,NY)")
    ap.add_argument("--override_years", default=None, help="e.g., 2024 or 2023-2024")
    ap.add_argument("--override_bundles", default=None, help="comma list of bundle ids (e.g., school,ssb_labeling)")
    args = ap.parse_args()

    if args.async_workers < 1:
        args.async_workers = 1
    if args.async_workers > 1 and not HAS_AIOHTTP:
        print("[warn] aiohttp not available; falling back to sequential getBill fetching.", file=sys.stderr)
        args.async_workers = 1
    if args.progress_interval <= 0:
        args.progress_interval = 0

    cfg = load_config(args.config)
    bundles = cfg.get("bundles", [])
    if args.override_bundles:
        wanted = set([b.strip() for b in args.override_bundles.split(",") if b.strip()])
        bundles = [b for b in bundles if b.get("id") in wanted]

    states = states_from_config(cfg)
    if args.override_states:
        states = [s.strip().upper() for s in args.override_states.split(",") if s.strip()]
    elif args.state_scope == "all":
        states = ["ALL"]

    if args.override_years:
        if "-" in args.override_years:
            a,b = args.override_years.split("-", 1)
            years = list(range(int(a), int(b)+1))
        else:
            years = [int(args.override_years)]
    else:
        years = years_from_config(cfg)

    ensure_dir(args.outdir)
    outdir_path = Path(args.outdir)
    directory_cache = build_directory_cache(outdir_path)

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

                cached_seen, cached_payloads = load_existing_cache(fpath)
                if cached_seen:
                    print(f"[info] reusing {len(cached_seen)} cached records for bundle={bundle_id} state={st} year={yr}")

                skip_log: List[Dict[str, Any]] = []

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
                            "note": "search-only output; parsing happens later",
                            "cache_reused": len(cached_seen),
                        }
                    }
                    fout.write(json.dumps(header, ensure_ascii=False) + "\n")

                    for raw in cached_payloads:
                        fout.write(raw + "\n")

                    seen = set(cached_seen)  # (bill_id, change_hash)
                    new_records_api = 0
                    global_reused_count = 0
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

                        new_targets: List[Tuple[int, str]] = []
                        for rec in results:
                            bill_id = rec.get("bill_id")
                            chash = rec.get("change_hash")
                            if not bill_id:
                                continue
                            key = (bill_id, chash)
                            if key in seen:
                                skip_log.append(
                                    {
                                        "bundle": bundle_id,
                                        "state": st,
                                        "year": yr,
                                        "bill_id": bill_id,
                                        "change_hash": chash,
                                    }
                                )
                                continue
                            new_targets.append((bill_id, chash))

                        if new_targets:
                            pending_targets: List[Tuple[int, str]] = []
                            for bill_id, chash in new_targets:
                                key = (bill_id, chash)
                                payload_str = directory_cache.get(key)
                                if payload_str:
                                    fout.write(payload_str + "\n")
                                    seen.add(key)
                                    global_reused_count += 1
                                    if args.progress_interval and global_reused_count % args.progress_interval == 0:
                                        print(
                                            f"[cache] bundle={bundle_id} state={st} year={yr} "
                                            f"reused_global={global_reused_count} last_bill={bill_id}"
                                        )
                                    continue
                                pending_targets.append((bill_id, chash))

                            fetched_results: Dict[int, Any] = {}
                            if pending_targets:
                                ordered_bill_ids: List[int] = []
                                seen_bill_ids: Set[int] = set()
                                for bid, _ in pending_targets:
                                    if bid not in seen_bill_ids:
                                        ordered_bill_ids.append(bid)
                                        seen_bill_ids.add(bid)

                                progress_state = {"fetched": 0, "total": len(ordered_bill_ids)}

                                def report_progress(bid: int, error: Optional[Exception] = None) -> None:
                                    progress_state["fetched"] += 1
                                    if args.progress_interval and progress_state["fetched"] % args.progress_interval == 0:
                                        status = "error" if error else "ok"
                                        print(
                                            f"[fetch] bundle={bundle_id} state={st} year={yr} "
                                            f"fetched={progress_state['fetched']}/{progress_state['total']} "
                                            f"last_bill={bid} status={status}"
                                        )

                                if args.async_workers > 1 and HAS_AIOHTTP:
                                    try:
                                        fetched_results = asyncio.run(
                                            fetch_bills_async(
                                                args.api_key,
                                                ordered_bill_ids,
                                                args.sleep,
                                                args.async_workers,
                                                progress_hook=report_progress,
                                            )
                                        )
                                    except Exception as exc:
                                        sys.stderr.write(f"[warn] async fetch failed, falling back to sequential: {exc}\n")
                                        args.async_workers = 1
                                        fetched_results = {}
                                        progress_state["fetched"] = 0

                                if args.async_workers == 1 or not fetched_results:
                                    if progress_state["total"] and progress_state["fetched"]:
                                        progress_state["fetched"] = 0
                                    for bid in ordered_bill_ids:
                                        time.sleep(args.sleep)
                                        try:
                                            fetched_results[bid] = get_bill(args.api_key, bid)
                                            report_progress(bid)
                                        except Exception as e:
                                            fetched_results[bid] = e
                                            report_progress(bid, error=e)

                                for bill_id, chash in pending_targets:
                                    payload = fetched_results.get(bill_id)
                                    if isinstance(payload, Exception):
                                        sys.stderr.write(f"[warn] getBill failed bill_id={bill_id}: {payload}\n")
                                        continue
                                    if not payload:
                                        sys.stderr.write(f"[warn] no payload for bill_id={bill_id}\n")
                                        continue
                                    payload_str = json.dumps(payload, ensure_ascii=False)
                                    fout.write(payload_str + "\n")
                                    directory_cache[(bill_id, chash)] = payload_str
                                    seen.add((bill_id, chash))
                                    new_records_api += 1

                        pc = _to_int(summary.get("page_current", 1), 1)
                        pt = _to_int(summary.get("page_total", 1), 1)

                        # optional: quick progress print
                        print(f"[pass] {bundle_id} {st} {yr} page {pc}/{pt} (+{len(results)} ids)")

                        if pc >= pt:
                            break
                        page += 1

                        time.sleep(args.sleep)

                print(
                    f"[ok] wrote {fpath}  (reused_local: {len(cached_seen)}, reused_global: {global_reused_count}, "
                    f"fetched: {new_records_api}, total unique: {len(seen)})"
                )
                for entry in skip_log:
                    print(
                        f"[cache-hit] bundle={entry['bundle']} state={entry['state']} year={entry['year']} "
                        f"bill={entry['bill_id']} change_hash={entry['change_hash']}"
                    )

if __name__ == "__main__":
    main()
