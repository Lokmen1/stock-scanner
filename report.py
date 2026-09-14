"""Report rendering, signal logging, and the forward-return evaluation harness."""
from __future__ import annotations

import datetime as dt
import re
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
    ("days_triggered", "Days", "{}", 5),
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
           flow_events: list | None = None, corp_flags: int = 0,
           crossings_today: dict | None = None,
           crossings_median: dict | None = None) -> str:
    total = sum(len(b) for b in buckets.values())
    out = [
        "=" * 92,
        f"  ANOMALY SCAN — {date:%A %d %B %Y}",
        f"  universe {n_universe:,} names   flagged {total}",
    ]
    if crossings_today is not None and crossings_median is not None:
        # How loose or tight today was BEFORE MAX_PER_BUCKET truncated the
        # list, against that gate's own trailing 90-day median. A quiet day is
        # quiet relative to its own recent distribution, not in absolute terms.
        labels = [("momentum", "momentum"), ("dislocation", "dislocation"),
                  ("high_participation_low_resolution", "high_participation_low_resolution")]
        parts = []
        for i, (key, label) in enumerate(labels):
            tag = "90d median" if i == 0 else "median"
            parts.append(f"{crossings_today.get(key, 0)} {label} "
                         f"({tag} {crossings_median.get(key, 0):.0f})")
        out.append("  crossings: " + " | ".join(parts))
    out += ["=" * 92, ""]
    if flow_events:
        # Tagged, not filtered — forced index/options flow is real supply-and-
        # demand information, just not fundamental information.
        out.append("  ** KNOWN MECHANICAL FLOW POSSIBLE TODAY **")
        for ev in flow_events:
            out.append(f"    - {ev}")
        out += ["  Volume signals today may be partly mechanical. Discount accordingly.", ""]
    if total == 0:
        out += ["  Nothing cleared the absolute thresholds today.", "",
                "  This is a valid and expected output. A scanner that always finds",
                "  twenty names has stopped distinguishing signal from Tuesday.", ""]

    titles = {
        "momentum": ("MOMENTUM ANOMALIES", "unusual upside on unusual volume"),
        "dislocation": ("DISLOCATION ANOMALIES", "unusual downside on unusual volume"),
        "mechanical_flow": (
            "MECHANICAL FLOW",
            "heavy volume, price has not moved — explained: a known index/options-"
            "flow day, or a large overnight gap typical of a corporate adjustment"),
        "unexplained_participation": (
            "UNEXPLAINED PARTICIPATION",
            "heavy volume, price has not moved, and nothing on the calendar "
            "explains it — could be accumulation, distribution, a block cross, "
            "or something not yet public"),
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
            "ret_z", "vol_z", "rel_ret", "gap", "breakout", "industry", "sic_division",
            "event_id", "event_age", "days_triggered", "max_score", "first_trigger_date",
            "mechanical_flow_possible",
            # Continuation features — deterministic, NOT inputs to score_day.
            "clv", "gap_retention", "rvol_1", "rvol_3", "rvol_5",
            "bo_20", "bo_60", "bo_252", "bo_all",
            "pre_event_compression", "float_turnover"]


_SYNTHETIC_TICKER = re.compile(r"^S\d{4}$")


def log_signals(buckets: dict, date: pd.Timestamp,
                 mechanical_flow_possible: bool = False) -> pd.DataFrame:
    """Append today's flags to the permanent signal log.

    This costs nothing and is the only thing here that produces real evidence.
    In six months it answers the one question that matters: does the score
    predict anything at all? It is also the forward-return data you would need
    to fit the weights instead of asserting them.

    `mechanical_flow_possible` tags rather than filters: forced index/options
    flow is real supply-and-demand information, just not fundamental
    information, so the day is logged and marked, never excluded.
    """
    rows = []
    for name, df in buckets.items():
        if df.empty:
            continue
        if df["ticker"].str.match(_SYNTHETIC_TICKER).any():
            raise ValueError(
                "Refusing to log synthetic-looking tickers (matches ^S\\d{4}$) "
                "to the real signal log. data/signals/ is timestamped evidence, "
                "not cache, and must never contain demo/synthetic entries."
            )
        d = df.copy()
        d["bucket"] = name
        d["mechanical_flow_possible"] = mechanical_flow_possible
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
def _cluster_key(df: pd.DataFrame) -> pd.Series:
    """(date x sector) — the unit that actually moves independently. 25 names
    triggering on one shared shock (one Nvidia print, one Fed day) are one
    cluster, not 25 observations."""
    industry = df["industry"] if "industry" in df.columns else pd.Series("Unknown", index=df.index)
    return df["date"].astype(str) + "|" + industry.fillna("Unknown")


def _cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray,
                          n_boot: int = 2000, seed: int = 7) -> tuple[float, float]:
    """95% CI for the mean, resampling CLUSTERS with replacement (not rows)."""
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in sampled])
        boot[i] = values[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _stats_lines(label: str, sub: pd.DataFrame, ret_col: str, mkt_col: str) -> list[str]:
    excess = (sub[ret_col] - sub[mkt_col]).to_numpy()
    clusters = _cluster_key(sub).to_numpy()
    n_clusters = len(np.unique(clusters))
    ci_lo, ci_hi = _cluster_bootstrap_ci(excess, clusters)
    ci_str = f"[{ci_lo:+.2%}, {ci_hi:+.2%}]" if not np.isnan(ci_lo) else "[too few clusters]"
    hit = (sub[ret_col] > sub[mkt_col]).mean()
    p5, p25, p75, p95 = np.percentile(excess, [5, 25, 75, 95])
    return [
        f"    {label:<22} n={len(sub):>4} (clusters={n_clusters:>3})  "
        f"mean {excess.mean():+.2%}  median {np.median(excess):+.2%}  "
        f"hit {hit:.0%}  95% CI {ci_str}",
        f"    {'':<22} pctl(excess): p5 {p5:+.2%}  p25 {p25:+.2%}  "
        f"p75 {p75:+.2%}  p95 {p95:+.2%}",
    ]


def _market_index(prices: pd.DataFrame) -> pd.Series:
    """Cross-sectional median daily return, compounded into an index level per
    date — the same market proxy used elsewhere, extended into a full path
    rather than single-horizon snapshots."""
    px = prices.sort_values(["ticker", "date"]).copy()
    px["ret_1d"] = px.groupby("ticker")["close"].pct_change()
    mkt_ret = px.groupby("date")["ret_1d"].median().sort_index()
    return (1 + mkt_ret.fillna(0)).cumprod()


def _excess_path_matrix(episodes: pd.DataFrame, prices: pd.DataFrame,
                         max_tau: int = C.EVENT_TIME_MAX_TAU) -> pd.DataFrame:
    """rows = episode (aligned to episodes.index), cols = tau (1..max_tau) ->
    cumulative excess return from the episode's first-trigger close, or NaN
    where fewer than tau future sessions exist yet."""
    mkt_index = _market_index(prices)
    trading_days = mkt_index.index
    close_wide = prices.pivot_table(index="date", columns="ticker", values="close")

    rows = []
    for idx, ep in episodes.iterrows():
        d0, t = ep["date"], ep["ticker"]
        base_close = close_wide.at[d0, t] if (d0 in close_wide.index and t in close_wide.columns) else np.nan
        base_mkt = mkt_index.get(d0, np.nan)
        if d0 not in trading_days or pd.isna(base_close) or pd.isna(base_mkt) or base_close == 0:
            rows.append([np.nan] * max_tau)
            continue
        pos0 = trading_days.get_loc(d0)
        path = []
        for tau in range(1, max_tau + 1):
            pos = pos0 + tau
            if pos >= len(trading_days):
                path.append(np.nan)
                continue
            d = trading_days[pos]
            c = close_wide.at[d, t] if t in close_wide.columns else np.nan
            if pd.isna(c):
                path.append(np.nan)
                continue
            path.append((c / base_close - 1) - (mkt_index.at[d] / base_mkt - 1))
        rows.append(path)
    return pd.DataFrame(rows, index=episodes.index, columns=range(1, max_tau + 1))


def _distribution_lines(label: str, values: pd.Series, clusters: pd.Series) -> list[str]:
    v = values.dropna()
    if v.empty:
        return [f"    {label:<22} n=   0  (no episodes with a full window yet)"]
    n_clusters = clusters.loc[v.index].nunique()
    p5, p25, p75, p95 = np.percentile(v, [5, 25, 75, 95])
    return [
        f"    {label:<22} n={len(v):>4} (clusters={n_clusters:>3})  "
        f"mean {v.mean():+.2%}  median {v.median():+.2%}",
        f"    {'':<22} pctl: p5 {p5:+.2%}  p25 {p25:+.2%}  p75 {p75:+.2%}  p95 {p95:+.2%}",
    ]


def _event_time_table(episodes: pd.DataFrame, path_matrix: pd.DataFrame,
                       max_tau: int = C.EVENT_TIME_MAX_TAU) -> list[str]:
    buckets = sorted(episodes["bucket"].unique())
    header = "  " + "tau".rjust(5) + "".join(b[:16].rjust(18) for b in buckets)
    lines = [header, "  " + "-" * (len(header) - 2)]
    idx_by_bucket = {b: episodes.index[episodes["bucket"] == b] for b in buckets}
    for tau in range(1, max_tau + 1):
        cells = [str(tau).rjust(5)]
        for b in buckets:
            vals = path_matrix.loc[idx_by_bucket[b], tau]
            m = vals.mean(skipna=True)
            cells.append(("—" if pd.isna(m) else f"{m:+.2%}").rjust(18))
        lines.append("  " + "".join(cells))
    return lines


def evaluate(prices: pd.DataFrame) -> str:
    """Join logged signals to subsequent prices and report forward returns, at
    both ticker-day and episode granularity, with cluster-robust CIs.

    Ticker-day rows are pseudo-replicated: a single shared shock (one earnings
    print moving a whole sector, one rebalance day) produces many correlated
    rows that look like independent evidence but aren't. Two mitigations:
      * episode level — one row per event_id (the first trigger), so a name
        that stays flagged for a week counts once, not seven times.
      * cluster bootstrap — CIs resample (date x sector) clusters, not rows,
        so correlated same-day/same-sector moves can't manufacture false
        precision.
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
    if "industry" not in m.columns:
        m["industry"] = "Unknown"

    episodes = m
    if "event_id" in m.columns and "first_trigger_date" in m.columns:
        episodes = m[m["date"] == m["first_trigger_date"]].drop_duplicates(subset="event_id")

    out = ["=" * 72, "  FORWARD-RETURN EVALUATION", "=" * 72, "",
           f"  signals logged: {len(log):,}   sessions: {log['date'].nunique()}",
           f"  episodes (first trigger only): {len(episodes):,}", "",
           "  CIs are bootstrapped over (date x sector) clusters, not rows — see",
           "  'clusters=' next to n. A wide or zero-straddling CI means too few",
           "  independent clusters to trust the point estimate, however large n looks.",
           ""]

    for level_name, frame in (("TICKER-DAY LEVEL", m),
                              ("EPISODE LEVEL (first trigger only)", episodes)):
        out.append(f"  === {level_name} ===")
        out.append("")
        for h in C.FORWARD_HORIZONS:
            c, mk = f"fwd_{h}", f"mkt_{h}"
            sub_h = frame.dropna(subset=[c, mk])
            if sub_h.empty:
                continue
            out.append(f"  --- {h}-day horizon ---")
            for b, grp in sub_h.groupby("bucket"):
                out += _stats_lines(b, grp, c, mk)
            top = sub_h.nlargest(max(len(sub_h) // 5, 1), "score")
            out += _stats_lines("top-quintile by score", top, c, mk)
            out.append("")

    # ---- path statistics (episode level only — see module docstring on why
    # ticker-day would re-introduce the pseudo-replication problem for a path
    # that's really one continuing situation, not several independent ones) ----
    path_matrix = _excess_path_matrix(episodes, prices)
    mfe = path_matrix.iloc[:, :C.MFE_MAE_WINDOW].max(axis=1, skipna=True)
    mae = path_matrix.iloc[:, :C.MFE_MAE_WINDOW].min(axis=1, skipna=True)
    clusters = _cluster_key(episodes)

    out.append(f"  === PATH STATISTICS (episode level, {C.MFE_MAE_WINDOW}-session window) ===")
    out.append("  MFE = best cumulative excess return reached at any point in the window.")
    out.append("  MAE = worst. Both relative to the episode's first-trigger close.")
    out.append("")
    for label, series in (("MFE", mfe), ("MAE", mae)):
        out.append(f"  --- {label} ---")
        for b in sorted(episodes["bucket"].unique()):
            idx = episodes.index[episodes["bucket"] == b]
            out += _distribution_lines(b, series.loc[idx], clusters)
        out.append("")

    out.append(f"  === EVENT-TIME CURVES (episode level, mean cumulative excess return) ===")
    out.append(f"  tau = sessions since first trigger. Sample thins out as tau grows — only")
    out.append(f"  episodes old enough to have {C.EVENT_TIME_MAX_TAU} future sessions reach the last row,")
    out.append("  so a bend near tau=60 can be a sample-composition artifact, not a real fade.")
    out.append("  This is what tells you whether a signal continues, fades, or reverses, and")
    out.append("  over what window it should actually be traded.")
    out.append("")
    out += _event_time_table(episodes, path_matrix)
    out.append("")

    out += ["  If top-quintile excess is not clearly above whole-basket excess,",
            "  the scoring layer is not earning its place and the weights are noise.", ""]
    return "\n".join(out)
