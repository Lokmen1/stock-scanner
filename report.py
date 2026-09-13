"""Report rendering, signal logging, and the forward-return evaluation harness."""
from __future__ import annotations

import datetime as dt
import pandas as pd
import numpy as np

import config as C


DISPLAY = [
    ("ticker", "Ticker", "{}", 7),
    ("close", "Price", "{:.2f}", 9),
    ("ret_1d", "Move", "{:+.1%}", 8),
    ("vol_z", "VolZ", "{:+.1f}", 6),
    ("ret_z", "RetZ", "{:+.1f}", 6),
    ("rel_ret", "vs Mkt", "{:+.1%}", 8),
    ("gap", "Gap", "{:+.1%}", 8),
    ("breakout_label", "B/O", "{}", 5),
    ("streak", "Days", "{}", 5),
    ("score", "Score", "{:.1f}", 7),
]


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "    (nothing cleared the threshold)\n"
    head = "  " + "".join(h.rjust(w) for _, h, _, w in DISPLAY)
    lines = [head, "  " + "-" * (len(head) - 2)]
    for _, r in df.iterrows():
        cells = []
        for col, _, fmt, w in DISPLAY:
            v = r.get(col)
            try:
                s = "—" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
            except (ValueError, TypeError):
                s = str(v)
            cells.append(s.rjust(w))
        lines.append("  " + "".join(cells))
    return "\n".join(lines) + "\n"


def render(buckets: dict, date: pd.Timestamp, n_universe: int,
           rebalance: bool = False, corp_flags: int = 0) -> str:
    total = sum(len(b) for b in buckets.values())
    out = [
        "=" * 92,
        f"  ANOMALY SCAN — {date:%A %d %B %Y}",
        f"  universe {n_universe:,} names   flagged {total}",
        "=" * 92, "",
    ]
    if rebalance:
        out += ["  ** QUARTERLY REBALANCE / TRIPLE WITCHING **",
                "  Volume signals today are largely mechanical. Discount accordingly.", ""]
    if total == 0:
        out += ["  Nothing cleared the absolute thresholds today.", "",
                "  This is a valid and expected output. A scanner that always finds",
                "  twenty names has stopped distinguishing signal from Tuesday.", ""]

    titles = {
        "momentum": ("MOMENTUM ANOMALIES", "unusual upside on unusual volume"),
        "dislocation": ("DISLOCATION ANOMALIES", "unusual downside on unusual volume"),
        "silent_accumulation": ("SILENT ACCUMULATION", "heavy volume, price has not moved — why?"),
    }
    for key, (title, sub) in titles.items():
        df = buckets.get(key, pd.DataFrame())
        out += [f"  {title}  ({len(df)})", f"  {sub}", ""]
        out.append(_table(df))
    if corp_flags:
        out += [f"  {corp_flags} name(s) suppressed as suspected unhandled corporate actions.", ""]
    out += [
        "-" * 92,
        "  NEXT STEP, for each name above: why did this happen?",
        "  Scores rank research priority. They are not probabilities and are not fitted.",
        "-" * 92, "",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ logging
LOG_COLS = ["date", "ticker", "bucket", "score", "close", "ret_1d",
            "ret_z", "vol_z", "rel_ret", "gap", "breakout", "streak"]


def log_signals(buckets: dict, date: pd.Timestamp) -> pd.DataFrame:
    """Append today's flags to the permanent signal log.

    This costs nothing and is the only thing here that produces real evidence.
    In six months it answers the one question that matters: does the score
    predict anything at all? It is also the forward-return data you would need
    to fit the weights instead of asserting them.
    """
    rows = []
    for name, df in buckets.items():
        if df.empty:
            continue
        d = df.copy()
        d["bucket"] = name
        rows.append(d[[c for c in LOG_COLS if c in d.columns]])
    if not rows:
        return pd.DataFrame(columns=LOG_COLS)
    log = pd.concat(rows, ignore_index=True)
    path = C.SIGNALS / f"{date:%Y-%m-%d}.parquet"
    log.to_parquet(path, index=False)
    return log


def load_signal_log() -> pd.DataFrame:
    files = sorted(C.SIGNALS.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


# ------------------------------------------------------------------ evaluate
def evaluate(prices: pd.DataFrame) -> str:
    """Join logged signals to subsequent prices and report forward returns.

    Two benchmarks matter:
      * the market over the same window — did the flag beat simply being long?
      * the flagged basket as a whole vs the top-scored subset — if the score
        adds nothing, the ranking layer is decoration.
    """
    log = load_signal_log()
    if log.empty:
        return "No logged signals yet. Run the scan daily; this becomes useful after ~6 months.\n"

    px = prices[["date", "ticker", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    for h in C.FORWARD_HORIZONS:
        px[f"fwd_{h}"] = px.groupby("ticker")["close"].shift(-h) / px["close"] - 1
    mkt = px.groupby("date")[[f"fwd_{h}" for h in C.FORWARD_HORIZONS]].median()
    mkt.columns = [f"mkt_{h}" for h in C.FORWARD_HORIZONS]

    log["date"] = pd.to_datetime(log["date"])
    m = log.merge(px, on=["date", "ticker"], how="left", suffixes=("", "_px"))
    m = m.merge(mkt, on="date", how="left")

    out = ["=" * 72, "  FORWARD-RETURN EVALUATION", "=" * 72, "",
           f"  signals logged: {len(log):,}   sessions: {log['date'].nunique()}", ""]

    for h in C.FORWARD_HORIZONS:
        c, mk = f"fwd_{h}", f"mkt_{h}"
        sub = m.dropna(subset=[c])
        if sub.empty:
            continue
        out.append(f"  --- {h}-day horizon  (n={len(sub):,}) ---")
        for b, grp in sub.groupby("bucket"):
            excess = (grp[c] - grp[mk]).mean()
            hit = (grp[c] > grp[mk]).mean()
            out.append(f"    {b:<22} mean {grp[c].mean():+.2%}   "
                       f"excess {excess:+.2%}   beat-mkt {hit:.0%}")
        top = sub.nlargest(max(len(sub) // 5, 1), "score")
        out.append(f"    {'top-quintile by score':<22} mean {top[c].mean():+.2%}   "
                   f"excess {(top[c] - top[mk]).mean():+.2%}")
        out.append("")

    out += ["  If top-quintile excess is not clearly above whole-basket excess,",
            "  the scoring layer is not earning its place and the weights are noise.", ""]
    return "\n".join(out)
