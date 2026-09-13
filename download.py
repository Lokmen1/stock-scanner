"""Data acquisition.

One grouped-daily call returns OHLCV for the entire US equity market for one
session, so a two-year backfill is ~500 calls rather than ~1.5m.

IMPORTANT — survivorship bias. Polygon's reference-ticker endpoint returns
tickers that are active TODAY. If you build the universe from that list and
pull history for it, every company that was delisted, acquired or went bankrupt
inside your window is silently absent. The downside tail of your anomaly
distribution is then drawn only from companies that survived, which is exactly
the tail the 'dislocation' bucket is supposed to find.

Two mitigations, both implemented here:
  1. Grouped-daily bars are stored for EVERY ticker that traded that day,
     before any universe filter. The price history is therefore survivorship-free.
  2. Universe membership is snapshotted daily to data/universe/YYYY-MM-DD.parquet
     and never overwritten, so from today forward you accumulate a genuine
     point-in-time universe.

What you cannot fix retroactively: universe membership BEFORE your first run.
Backtests over the backfilled period remain contaminated. Treat them as
indicative only, and treat forward-logged signals as the real evidence.
"""
from __future__ import annotations

import time
import datetime as dt
from typing import Iterable

import pandas as pd
import requests

import config as C


def _get(url: str, params: dict | None = None) -> dict:
    params = dict(params or {})
    params["apiKey"] = C.POLYGON_KEY
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:                       # rate limited
            time.sleep(2 ** attempt * 5)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"repeated rate-limit on {url}")


def grouped_daily(date: dt.date) -> pd.DataFrame:
    """OHLCV for every US equity on one session.

    adjusted=true makes Polygon apply split and dividend adjustment. This is not
    optional: on unadjusted data a 3-for-1 split reads as a -67% return on huge
    volume and will dominate your dislocation bucket.
    """
    url = f"{C.POLYGON_BASE}/v2/aggs/grouped/locale/us/market/stocks/{date:%Y-%m-%d}"
    js = _get(url, {"adjusted": "true"})
    rows = js.get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "T": "ticker", "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap", "n": "trades",
    })
    df["date"] = pd.Timestamp(date)
    keep = ["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "trades"]
    return df[[c for c in keep if c in df.columns]]


def backfill(start: dt.date, end: dt.date, skip_existing: bool = True) -> None:
    """Store one parquet file per session. Idempotent."""
    day = start
    while day <= end:
        if day.weekday() < 5:                          # crude; holidays return empty
            path = C.PRICES / f"{day:%Y-%m-%d}.parquet"
            if not (skip_existing and path.exists()):
                df = grouped_daily(day)
                if not df.empty:
                    df.to_parquet(path, index=False)
                    print(f"  {day} {len(df):>6} tickers")
                time.sleep(C.REQUEST_SLEEP)
        day += dt.timedelta(days=1)


def reference_tickers() -> pd.DataFrame:
    """Active common stock on the three main exchanges, with share counts where available."""
    out, url = [], f"{C.POLYGON_BASE}/v3/reference/tickers"
    params = {"market": "stocks", "type": "CS", "active": "true", "limit": 1000}
    while True:
        js = _get(url, params)
        out.extend(js.get("results", []))
        nxt = js.get("next_url")
        if not nxt:
            break
        url, params = nxt, {}
        time.sleep(C.REQUEST_SLEEP)

    df = pd.DataFrame(out)
    if df.empty:
        return df
    cols = [c for c in ("ticker", "name", "type", "primary_exchange",
                        "share_class_shares_outstanding",
                        "weighted_shares_outstanding", "sic_code") if c in df.columns]
    df = df[cols]
    df = df[df["primary_exchange"].isin(C.ALLOWED_EXCHANGES)]
    shares = None
    for c in ("weighted_shares_outstanding", "share_class_shares_outstanding"):
        if c in df.columns:
            shares = df[c] if shares is None else shares.fillna(df[c])
    df["shares_outstanding"] = shares
    df["snapshot_date"] = pd.Timestamp(dt.date.today())
    return df.reset_index(drop=True)


def snapshot_universe() -> pd.DataFrame:
    """Write today's universe membership. Never overwrite an earlier snapshot."""
    df = reference_tickers()
    path = C.UNIVERSE / f"{dt.date.today():%Y-%m-%d}.parquet"
    if not path.exists():
        df.to_parquet(path, index=False)
    return df


def load_prices(days: int | None = None, prices_dir=None) -> pd.DataFrame:
    """Load stored sessions into one frame, most recent `days` sessions."""
    prices_dir = prices_dir or C.PRICES
    files = sorted(prices_dir.glob("*.parquet"))
    if days:
        files = files[-days:]
    if not files:
        raise FileNotFoundError(
            "No price data. Run `python main.py backfill` first, "
            "or `python main.py demo` to generate synthetic data."
        )
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def latest_universe(universe_dir=None) -> pd.DataFrame:
    universe_dir = universe_dir or C.UNIVERSE
    files = sorted(universe_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])
