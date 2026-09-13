#!/usr/bin/env python3
"""Anomaly scanner — CLI.

    python main.py demo              generate synthetic data and run (no API key)
    python main.py backfill --days N pull N sessions of real data from Polygon
    python main.py universe          snapshot today's point-in-time universe
    python main.py scan              run the scan, print and save the report
    python main.py evaluate          forward-return analysis of logged signals
    python main.py diagnose          feature correlations and threshold hit rates
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

import config as C
import features as F
import rank as R
import report as REP


def cmd_scan(args) -> None:
    import download as D
    prices = D.load_prices(days=C.HISTORY_DAYS)
    ref = D.latest_universe()

    feats = F.compute(prices)
    inv = R.apply_universe(feats, ref if not ref.empty else None)

    last = inv["date"].max()
    today = inv[inv["date"] == last]
    n_universe = today["ticker"].nunique()
    corp_flags = int(today["suspect_corp_action"].sum())

    scored = R.score_day(today)
    buckets = R.bucket(scored)
    buckets = R.add_persistence(buckets, REP.load_signal_log())

    text = REP.render(buckets, last, n_universe,
                      rebalance=R.is_rebalance_day(last), corp_flags=corp_flags)
    print(text)

    path = C.REPORTS / f"{last:%Y-%m-%d}.txt"
    path.write_text(text)
    REP.log_signals(buckets, last)
    print(f"  saved: {path}")


def cmd_evaluate(args) -> None:
    import download as D
    print(REP.evaluate(D.load_prices()))


def cmd_diagnose(args) -> None:
    import download as D
    prices = D.load_prices(days=C.HISTORY_DAYS)
    feats = F.compute(prices)
    ref = D.latest_universe()
    inv = R.apply_universe(feats, ref if not ref.empty else None)
    recent = inv[inv["date"] >= inv["date"].max() - pd.Timedelta(days=90)]

    print("\n  FEATURE CORRELATION (Spearman)")
    print("  Off-diagonals above ~0.7 mean you are weighting one signal twice.\n")
    print(F.feature_correlations(recent).round(2).to_string())

    print("\n  DAILY FLAG COUNTS (how often does each gate fire?)\n")
    per_day = []
    for d, g in recent.groupby("date"):
        b = R.bucket(R.score_day(g))
        per_day.append({k: len(v) for k, v in b.items()})
    pd_df = pd.DataFrame(per_day)
    print(pd_df.describe().round(1).to_string())
    zero = (pd_df.sum(axis=1) == 0).mean()
    print(f"\n  sessions with NO flags: {zero:.0%}")
    print("  If this is 0%, your gates are too loose to ever say 'nothing today'.")
    print("  If above ~40%, they are too tight to generate research flow.\n")


def cmd_backfill(args) -> None:
    import download as D
    if not C.POLYGON_KEY:
        sys.exit("Set POLYGON_API_KEY in your environment first.")
    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=int(args.days * 1.45))   # calendar vs trading days
    print(f"  backfilling {start} → {end}")
    D.backfill(start, end)


def cmd_universe(args) -> None:
    import download as D
    if not C.POLYGON_KEY:
        sys.exit("Set POLYGON_API_KEY in your environment first.")
    df = D.snapshot_universe()
    print(f"  snapshotted {len(df):,} common stocks")


# --------------------------------------------------------------------- demo
def cmd_demo(args) -> None:
    """Synthetic market so the pipeline can be exercised without an API key.

    Deliberately plants a handful of known anomalies so you can verify the
    scanner finds what it is supposed to find before trusting it on real data.
    """
    rng = np.random.default_rng(7)
    n_names = 600
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=320)
    n_days = len(dates)          # bdate_range trims if today is a weekend
    tickers = [f"S{i:04d}" for i in range(n_names)]

    # heterogeneous vol so the per-name standardisation has work to do
    vol = rng.uniform(0.010, 0.055, n_names)
    px0 = rng.uniform(8, 260, n_names)
    liq = rng.lognormal(16.0, 1.1, n_names)
    liq[:10] = np.exp(18.0)          # plants must clear the $5m ADV gate

    frames = []
    for i, t in enumerate(tickers):
        r = rng.normal(0.0003, vol[i], n_days)
        # quarterly earnings jumps: inflate naive sigma, which robust scale ignores
        for j in range(45, n_days, 63):
            r[j] += rng.normal(0, vol[i] * 5)
        close = px0[i] * np.exp(np.cumsum(r))
        v = liq[i] * np.exp(rng.normal(0, 0.45, n_days)) / close
        frames.append(pd.DataFrame({
            "date": dates, "ticker": t, "close": close, "volume": v,
            "open": close * (1 + rng.normal(0, vol[i] * 0.3, n_days)),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.01
    df["low"] = df[["open", "close"]].min(axis=1) * 0.99

    # ---- plant known anomalies on the final session ----
    last = dates[-1]
    plants = {
        "S0001": ("momentum", 0.14, 9.0),
        "S0002": ("momentum", 0.09, 6.0),
        "S0003": ("dislocation", -0.17, 11.0),
        "S0004": ("dislocation", -0.11, 7.0),
        "S0005": ("silent", 0.004, 14.0),     # the interesting one
        "S0006": ("silent", -0.002, 12.0),
        "S0007": ("corp_action", 0.62, 1.0),  # should be SUPPRESSED
    }
    for t, (_, ret, volmult) in plants.items():
        m = (df["ticker"] == t) & (df["date"] == last)
        prev = df[(df["ticker"] == t) & (df["date"] == dates[-2])]["close"].iloc[0]
        df.loc[m, "close"] = prev * (1 + ret)
        df.loc[m, "open"] = prev * (1 + ret * 0.6)
        if volmult == 1.0:           # corp-action plant: pin volume to typical
            typ = df[(df["ticker"] == t)]["volume"].median()
            df.loc[m, "volume"] = typ
        else:
            df.loc[m, "volume"] *= volmult

    for d, g in df.groupby("date"):
        g.to_parquet(C.PRICES / f"{d:%Y-%m-%d}.parquet", index=False)

    ref = pd.DataFrame({
        "ticker": tickers, "type": "CS", "primary_exchange": "XNAS",
        "shares_outstanding": rng.uniform(3e7, 9e8, n_names),
        "snapshot_date": pd.Timestamp.today().normalize(),
    })
    ref.to_parquet(C.UNIVERSE / f"{dt.date.today():%Y-%m-%d}.parquet", index=False)

    print(f"  generated {n_days} sessions × {n_names} names")
    print("  planted: S0001-2 momentum | S0003-4 dislocation | "
          "S0005-6 silent | S0007 corp-action (should be suppressed)\n")
    cmd_scan(args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan").set_defaults(fn=cmd_scan)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    sub.add_parser("diagnose").set_defaults(fn=cmd_diagnose)
    sub.add_parser("universe").set_defaults(fn=cmd_universe)
    b = sub.add_parser("backfill")
    b.add_argument("--days", type=int, default=520)
    b.set_defaults(fn=cmd_backfill)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
