"""SIC code lookup, via SEC rather than Polygon.

Polygon's bulk reference-tickers endpoint (used for daily universe snapshots)
does not return sic_code at all — confirmed empty on a live sample. The
per-ticker Polygon endpoint does return it, but that's ~5,300 individual calls
for the full universe, hours at this project's rate-limited tier.

SEC is the alternative: one bulk file maps ticker -> CIK, then one call per
CIK against data/submissions returns sic/sicDescription directly. At SEC's
~10 req/sec guidance that's minutes, not hours, for the whole universe — cheap
enough that a full bulk zip archive (SEC also publishes
Archives/edgar/daily-index/bulkdata/submissions.zip, ~1.5GB, every CIK ever
registered) isn't worth the bandwidth/disk for the ~5,300 tickers we need.

A persistent requests.Session is used for the per-CIK loop — a fresh
connection (TCP + TLS) per call was the difference between this taking
minutes and taking the better part of an hour. Progress is printed with
flush=True and the cache is checkpointed to disk every CHECKPOINT tickers, so
a kill (or crash) never loses more than one checkpoint's worth of work.

Coverage is inherently partial: foreign issuers filing 20-F, some ADRs, and
anything not SEC-registered have no CIK and are left "Unknown" rather than
guessed.

Fetching from SEC never happens on the daily scan path — only `update_cache`
does, and only main.py's `sic-update` command calls that. `attach_sic` (used
by cmd_scan) is a pure local read + merge, no network access.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

import config as C

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
CHECKPOINT = 100


def _headers() -> dict:
    return {"User-Agent": C.SEC_USER_AGENT}


def fetch_cik_map(session: requests.Session | None = None) -> dict[str, str]:
    """ticker -> zero-padded 10-digit CIK string, from SEC's one bulk file."""
    get = session.get if session else requests.get
    r = get(TICKERS_URL, headers=_headers(), timeout=30)
    r.raise_for_status()
    rows = r.json().values()
    return {row["ticker"].upper(): f"{row['cik_str']:0>10}" for row in rows}


def fetch_sic(cik: str, session: requests.Session | None = None) -> tuple[int | None, str | None]:
    """(sic_code, sic_description) for one CIK, or (None, None) if missing."""
    get = session.get if session else requests.get
    r = get(SUBMISSIONS_URL.format(cik=cik), headers=_headers(), timeout=30)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    js = r.json()
    sic = js.get("sic")
    return (int(sic) if sic else None), js.get("sicDescription")


def load_cache() -> pd.DataFrame:
    if not C.SIC_MAP.exists():
        return pd.DataFrame(columns=["ticker", "cik", "sic_code", "sic_description", "fetched_at"])
    return pd.read_parquet(C.SIC_MAP)


def _merge_and_save(cache: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(new_rows)
    merged = pd.concat([cache[~cache["ticker"].isin(new["ticker"])], new], ignore_index=True)
    merged.to_parquet(C.SIC_MAP, index=False)
    return merged


def update_cache(tickers: list[str], force_refresh_days: int = C.SIC_REFRESH_DAYS) -> pd.DataFrame:
    """Fetch SIC codes for `tickers` not yet cached, or for everyone if the
    cache is older than `force_refresh_days`. Network calls happen ONLY here —
    call this from a deliberate maintenance step, never from cmd_scan.

    Checkpoints to disk every CHECKPOINT tickers and prints live throughput/
    ETA. Prints a final coverage summary (fraction of `tickers` with a
    resolved SIC).
    """
    cache = load_cache()
    tickers = sorted(set(t.upper() for t in tickers))

    stale = (cache.empty or cache["fetched_at"].max() <
             pd.Timestamp.now() - pd.Timedelta(days=force_refresh_days))
    need = tickers if stale else [t for t in tickers if t not in set(cache["ticker"])]

    if not need:
        print(f"  SIC cache up to date ({len(cache)} tickers, refreshed within "
              f"{force_refresh_days}d). Nothing to fetch.", flush=True)
        return cache

    print("  fetching CIK map...", flush=True)
    session = requests.Session()
    session.headers.update(_headers())
    cik_map = fetch_cik_map(session=session)
    have_cik = [t for t in need if t in cik_map]
    no_cik = [t for t in need if t not in cik_map]
    print(f"  {len(have_cik)}/{len(need)} tickers resolve to a CIK "
          f"(the rest are likely foreign issuers/ADRs without SEC registration)", flush=True)

    now = pd.Timestamp.now()
    # No-CIK tickers get a row now (sic_code=NaN) so they aren't re-attempted
    # every run — they're not coming back without a CIK.
    pending = [{"ticker": t, "cik": None, "sic_code": None, "sic_description": None,
                "fetched_at": now} for t in no_cik]

    delay = 1.0 / C.SEC_REQUESTS_PER_SEC
    start = time.monotonic()

    for i, t in enumerate(have_cik):
        cik = cik_map[t]
        sic, desc = fetch_sic(cik, session=session)
        pending.append({"ticker": t, "cik": cik, "sic_code": sic,
                         "sic_description": desc, "fetched_at": now})
        time.sleep(delay)

        done = i + 1
        if done % CHECKPOINT == 0 or done == len(have_cik):
            cache = _merge_and_save(cache, pending)
            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed > 0 else 0
            remaining_min = (len(have_cik) - done) / rate / 60 if rate > 0 else float("inf")
            print(f"  {done}/{len(have_cik)} fetched ({rate:.1f}/s, "
                  f"~{remaining_min:.1f} min left) — checkpoint saved", flush=True)

    resolved = cache[cache["ticker"].isin(tickers)]["sic_code"].notna().sum()
    print(f"  coverage: {resolved}/{len(tickers)} tickers ({resolved / len(tickers):.0%}) "
          f"have a SIC code", flush=True)
    return cache


def attach_sic(ref: pd.DataFrame) -> pd.DataFrame:
    """Left-merge cached sic_code onto `ref` by ticker. Pure local read, no
    network access — safe to call on the daily scan path."""
    cache = load_cache()
    if cache.empty or ref.empty:
        return ref
    ref = ref.copy()
    sic = cache.set_index("ticker")["sic_code"]
    ref["sic_code"] = ref["ticker"].str.upper().map(sic)
    return ref
