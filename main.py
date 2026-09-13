#!/usr/bin/env python3
"""Anomaly scanner — CLI.

    python main.py demo              generate synthetic data and run (no API key)
    python main.py backfill --days N pull N sessions of real data from Polygon
    python main.py universe          snapshot today's point-in-time universe
    python main.py scan              run the scan, print and save the report
    python main.py evaluate          forward-return analysis of logged signals
    python main.py diagnose          feature correlations and threshold hit rates
    python main.py sic-update        refresh the SEC SIC-code cache (not on the scan path)
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys

import numpy as np
import pandas as pd

import config as C
import episodes as EP
import features as F
import flow_calendar as FLOW
import rank as R
import report as REP
import sic as SIC


def cmd_scan(args, prices_dir=None, universe_dir=None, log=True) -> None:
    """`prices_dir`/`universe_dir`/`log` let `demo` route through the synthetic
    sandbox instead of real data — real `scan` invocations never pass these."""
    import download as D
    prices = D.load_prices(days=C.HISTORY_DAYS, prices_dir=prices_dir)
    ref = D.latest_universe(universe_dir=universe_dir)
    ref = SIC.attach_sic(ref) if not ref.empty else ref

    feats = F.compute(prices)
    inv = R.apply_universe(feats, ref if not ref.empty else None)

    last = inv["date"].max()
    today = inv[inv["date"] == last]
    n_universe = today["ticker"].nunique()
    corp_flags = int(today["suspect_corp_action"].sum())

    crossings_today = R.raw_crossings(today)
    hist90 = inv[inv["date"] >= last - pd.Timedelta(days=90)]
    hist_crossings = pd.DataFrame(
        [R.raw_crossings(g) for _, g in hist90.groupby("date")])
    crossings_median = hist_crossings.median().to_dict()

    scored = R.score_day(today)
    buckets = R.bucket(scored)
    history = REP.load_signal_log() if log else pd.DataFrame()
    trading_days = pd.DatetimeIndex(sorted(inv["date"].unique()))
    buckets = {k: EP.assign_episodes(v, history, trading_days, last)
               for k, v in buckets.items()}

    flow_events = FLOW.known_flow_events(last)
    text = REP.render(buckets, last, n_universe,
                      flow_events=flow_events, corp_flags=corp_flags,
                      crossings_today=crossings_today, crossings_median=crossings_median)
    print(text)

    path = C.REPORTS / f"{last:%Y-%m-%d}.txt"
    path.write_text(text)
    if log:
        REP.log_signals(buckets, last, mechanical_flow_possible=bool(flow_events))
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


def cmd_sic_update(args) -> None:
    """Refresh the local SEC SIC-code cache. Makes network calls to SEC — by
    design this is its own command, never invoked from the scan path."""
    import download as D
    ref = D.latest_universe()
    if ref.empty:
        sys.exit("No universe snapshot found. Run `python main.py universe` first.")
    SIC.update_cache(ref["ticker"].tolist())


# --------------------------------------------------------------------- demo
def cmd_demo(args) -> None:
    """Synthetic market so the pipeline can be exercised without an API key.

    Deliberately plants a handful of known anomalies so you can verify the
    scanner finds what it is supposed to find before trusting it on real data.

    Writes only into data/demo_prices/ and data/demo_universe/ — never the
    real data/prices/, data/universe/, or data/signals/ (the signal log is
    timestamped evidence and must never contain synthetic entries).
    """
    # Wipe first so stale synthetic sessions from a previous run (e.g. a demo
    # run on an earlier calendar day, which shifts the whole date range) can't
    # accumulate alongside this run's files.
    shutil.rmtree(C.DEMO_PRICES, ignore_errors=True)
    shutil.rmtree(C.DEMO_UNIVERSE, ignore_errors=True)
    C.DEMO_PRICES.mkdir(parents=True, exist_ok=True)
    C.DEMO_UNIVERSE.mkdir(parents=True, exist_ok=True)

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
    # Magnitudes are derived from the live gate thresholds in config.py, not
    # hard-coded, so this smoke test stays valid however the gates are tuned.
    # Each plant's return/volume is solved from that TICKER'S OWN median/MAD
    # history (the exact same calculation features.py uses), landing it ~1.5x
    # its relevant threshold in z-terms — comfortably over the gate, not just
    # barely. High-participation/low-resolution return targets stay well
    # UNDER the quiet ceiling instead, since that gate runs the other way.
    last = dates[-1]
    MARGIN = 1.5

    def hist_med_scale(s: pd.Series) -> tuple[float, float]:
        """Median/MAD-scale of a ticker's own pre-plant history, exactly as
        features.py computes it, evaluated at the plant day (final row)."""
        hist = s.shift(1)
        med = hist.rolling(C.LOOKBACK, min_periods=C.MIN_OBS).median()
        mad = (hist - med).abs().rolling(C.LOOKBACK, min_periods=C.MIN_OBS).median()
        return med.iloc[-1], mad.iloc[-1] * C.MAD_SCALE

    def solve_plant(t: str, z_ret: float, z_vol: float) -> tuple[float, float]:
        """Return (close, volume) on the plant day that lands at the given
        target z-scores given this ticker's own history."""
        sub = df[df["ticker"] == t].sort_values("date")
        prev = sub[sub["date"] == dates[-2]]["close"].iloc[0]
        ret_med, ret_scale = hist_med_scale(sub["close"].pct_change())
        dv_med, dv_scale = hist_med_scale(np.log((sub["close"] * sub["volume"]).clip(lower=1.0)))
        close = prev * (1 + ret_med + z_ret * ret_scale)
        dollar_volume = np.exp(dv_med + z_vol * dv_scale)
        return close, dollar_volume / close

    plant_targets = {
        "S0001": (MARGIN * C.Z_RETURN_GATE, MARGIN * C.Z_VOLUME_GATE),
        "S0002": (MARGIN * C.Z_RETURN_GATE, MARGIN * C.Z_VOLUME_GATE),
        "S0003": (-MARGIN * C.Z_RETURN_GATE, MARGIN * C.Z_VOLUME_GATE),
        "S0004": (-MARGIN * C.Z_RETURN_GATE, MARGIN * C.Z_VOLUME_GATE),
        "S0005": (C.Z_RETURN_QUIET * 0.4, MARGIN * C.Z_VOLUME_STRONG),     # the interesting one
        "S0006": (-C.Z_RETURN_QUIET * 0.4, MARGIN * C.Z_VOLUME_STRONG),
    }
    for t, (z_ret, z_vol) in plant_targets.items():
        close, volume = solve_plant(t, z_ret, z_vol)
        prev = df[(df["ticker"] == t) & (df["date"] == dates[-2])]["close"].iloc[0]
        m = (df["ticker"] == t) & (df["date"] == last)
        df.loc[m, "close"] = close
        df.loc[m, "open"] = prev * (1 + (close / prev - 1) * 0.6)
        df.loc[m, "volume"] = volume

    # Corp-action plant: big move on ORDINARY volume — should be suppressed.
    t = "S0007"
    prev = df[(df["ticker"] == t) & (df["date"] == dates[-2])]["close"].iloc[0]
    ret = MARGIN * C.CORP_ACTION_RET_GUARD
    m = (df["ticker"] == t) & (df["date"] == last)
    df.loc[m, "close"] = prev * (1 + ret)
    df.loc[m, "open"] = prev * (1 + ret * 0.6)
    df.loc[m, "volume"] = df[df["ticker"] == t]["volume"].median()

    for d, g in df.groupby("date"):
        g.to_parquet(C.DEMO_PRICES / f"{d:%Y-%m-%d}.parquet", index=False)

    ref = pd.DataFrame({
        "ticker": tickers, "type": "CS", "primary_exchange": "XNAS",
        "shares_outstanding": rng.uniform(3e7, 9e8, n_names),
        "snapshot_date": pd.Timestamp.today().normalize(),
    })
    ref.to_parquet(C.DEMO_UNIVERSE / f"{dt.date.today():%Y-%m-%d}.parquet", index=False)

    print(f"  generated {n_days} sessions × {n_names} names")
    print("  planted: S0001-2 momentum | S0003-4 dislocation | "
          "S0005-6 high-participation | S0007 corp-action (should be suppressed)\n")
    cmd_scan(args, prices_dir=C.DEMO_PRICES, universe_dir=C.DEMO_UNIVERSE, log=False)
    _print_plant_check()


def _print_plant_check() -> None:
    """Show each planted anomaly's ret_z/vol_z against the live gates, so a
    pass/fail is visible with its margin, not just as a name on the report."""
    import download as D
    prices = D.load_prices(days=C.HISTORY_DAYS, prices_dir=C.DEMO_PRICES)
    feats = F.compute(prices)
    ref = D.latest_universe(universe_dir=C.DEMO_UNIVERSE)
    inv = R.apply_universe(feats, ref if not ref.empty else None)
    today = inv[inv["date"] == inv["date"].max()].set_index("ticker")

    print("  PLANTED ANOMALY CHECK  (ret_z / vol_z vs current gates)\n")
    print(f"  {'Ticker':<7}{'Bucket':<13}{'RetZ':>8}  {'gate':<10}{'VolZ':>8}  {'gate':<10}Result")
    print("  " + "-" * 68)

    z_checks = [
        ("S0001", "momentum", ">=", C.Z_RETURN_GATE),
        ("S0002", "momentum", ">=", C.Z_RETURN_GATE),
        ("S0003", "dislocation", "<=", -C.Z_RETURN_GATE),
        ("S0004", "dislocation", "<=", -C.Z_RETURN_GATE),
        ("S0005", "hplr", "abs<=", C.Z_RETURN_QUIET),
        ("S0006", "hplr", "abs<=", C.Z_RETURN_QUIET),
    ]
    for t, bucket, ret_op, ret_gate in z_checks:
        row = today.loc[t]
        ret_z, vol_z = row["ret_z"], row["vol_z"]
        vol_gate = C.Z_VOLUME_STRONG if bucket == "hplr" else C.Z_VOLUME_GATE
        ret_ok = abs(ret_z) <= ret_gate if ret_op == "abs<=" else (
            ret_z >= ret_gate if ret_op == ">=" else ret_z <= ret_gate)
        vol_ok = vol_z >= vol_gate
        result = "PASS" if (ret_ok and vol_ok) else "FAIL"
        ret_gate_label = f"|.|<={ret_gate:.1f}" if ret_op == "abs<=" else f"{ret_op}{ret_gate:.1f}"
        vol_gate_label = f">={vol_gate:.1f}"
        print(f"  {t:<7}{bucket:<13}{ret_z:>+8.1f}  {ret_gate_label:<10}"
              f"{vol_z:>+8.1f}  {vol_gate_label:<10}{result}")

    row = today.loc["S0007"]
    ret_1d, vol_z = row["ret_1d"], row["vol_z"]
    suppressed = bool(row["suspect_corp_action"])
    print(f"  {'S0007':<7}{'corp-action':<13}{ret_1d:>+7.1%}  "
          f"{'>' + format(C.CORP_ACTION_RET_GUARD, '.2f'):<10}"
          f"{vol_z:>+8.1f}  {'<' + format(C.CORP_ACTION_VOL_FLOOR, '.1f'):<10}"
          f"{'SUPPRESSED' if suppressed else 'NOT SUPPRESSED'}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan").set_defaults(fn=cmd_scan)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    sub.add_parser("diagnose").set_defaults(fn=cmd_diagnose)
    sub.add_parser("universe").set_defaults(fn=cmd_universe)
    sub.add_parser("sic-update").set_defaults(fn=cmd_sic_update)
    b = sub.add_parser("backfill")
    b.add_argument("--days", type=int, default=520)
    b.set_defaults(fn=cmd_backfill)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
