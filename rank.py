"""Universe filtering, scoring and bucketing."""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd

import config as C


# ------------------------------------------------------------------ universe
def apply_universe(df: pd.DataFrame, ref: pd.DataFrame | None) -> pd.DataFrame:
    """Liquidity and quality gates. Returns the investable slice only."""
    out = df[
        (df["close"] >= C.MIN_PRICE)
        & (df["adv_20"] >= C.MIN_ADV_20)
    ].copy()

    if ref is not None and not ref.empty:
        ok = set(ref["ticker"])
        out = out[out["ticker"].isin(ok)]
        if "shares_outstanding" in ref.columns:
            sh = ref.set_index("ticker")["shares_outstanding"]
            out["market_cap"] = out["close"] * out["ticker"].map(sh)
            # Only exclude where we actually KNOW the cap is too small.
            # Missing share counts must not silently delete the name.
            out = out[~(out["market_cap"] < C.MIN_MARKET_CAP)]
    else:
        out["market_cap"] = np.nan

    # Structural ticker junk that slips through type filters
    bad = out["ticker"].str.contains(r"[.\-]W$|[.\-]U$|[.\-]R$|WS$", regex=True, na=False)
    return out[~bad]


# ------------------------------------------------------------------ scoring
def score_day(today: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks, then a weighted blend.

    Percentile ranks rather than raw z-scores: return distributions have fat
    tails, so in a weighted sum of z-scores a single z=15 observation swamps
    every other input. Winsorising patches that; ranking avoids it.

    The output is an ATTENTION score — how much research time a name deserves
    today — not a calibrated probability of a subsequent move.
    """
    d = today.copy()
    d["return_rank"] = d["ret_z"].abs().rank(pct=True)
    d["volume_rank"] = d["vol_z"].rank(pct=True)
    d["relative_rank"] = d["rel_ret"].abs().rank(pct=True)
    d["gap_rank"] = d["gap_z"].abs().rank(pct=True)
    d["breakout_rank"] = d["breakout"].rank(pct=True)

    # One event should cast one vote. A stock that gaps +6% and closes +7%
    # scores on return, gap AND relative-strength for a single piece of news.
    d["price_rank"] = d[["return_rank", "relative_rank", "gap_rank"]].max(axis=1)

    weights = (C.SCORE_WEIGHTS_ORTHOGONAL if C.SCORE_MODE == "orthogonal"
               else C.SCORE_WEIGHTS)
    d["score"] = sum(d[k].fillna(0.5) * w for k, w in weights.items()) * 100
    return d


# ------------------------------------------------------------------ buckets
def bucket(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Three behaviourally distinct lists, each behind an ABSOLUTE gate.

    The absolute gates are the point. A pure top-N ranking emits N names every
    evening whether or not anything happened, so on a quiet Tuesday the list is
    noise wearing the same format as a real signal. With gates, a quiet day
    returns a short list — or none — and the report stays trustworthy.
    """
    d = d[~d["suspect_corp_action"]]

    momentum = d[(d["ret_z"] >= C.Z_RETURN_GATE) & (d["vol_z"] >= C.Z_VOLUME_GATE)]
    dislocation = d[(d["ret_z"] <= -C.Z_RETURN_GATE) & (d["vol_z"] >= C.Z_VOLUME_GATE)]
    # Heavy volume, little price movement: someone is transacting size and the
    # price has not resolved it yet. The only one of the three that is not
    # already visible on every free screener.
    silent = d[(d["vol_z"] >= C.Z_VOLUME_STRONG) & (d["ret_z"].abs() <= C.Z_RETURN_QUIET)]

    return {
        "momentum": momentum.nlargest(C.MAX_PER_BUCKET, "score"),
        "dislocation": dislocation.nlargest(C.MAX_PER_BUCKET, "score"),
        "silent_accumulation": silent.nlargest(C.MAX_PER_BUCKET, "vol_z"),
    }


# ------------------------------------------------------------------ context
def add_persistence(buckets: dict, history: pd.DataFrame, days: int = 5) -> dict:
    """How many of the last `days` sessions did this name already appear?

    A name on the list three days running is a different object from a one-day
    spike — usually a developing situation rather than a single print.
    """
    if history.empty:
        for b in buckets.values():
            b["streak"] = 1
        return buckets
    cutoff = history["date"].max() - pd.Timedelta(days=days * 2)
    recent = history[history["date"] >= cutoff]
    counts = recent.groupby("ticker").size()
    for b in buckets.values():
        b["streak"] = b["ticker"].map(counts).fillna(0).astype(int) + 1
    return buckets


def is_rebalance_day(date: pd.Timestamp) -> bool:
    """Third Friday of Mar/Jun/Sep/Dec — quarterly index rebalance and triple
    witching. Enormous volume with no fundamental content. Nokia's Euro Stoxx 50
    inclusion is the same class of event: real volume, nothing to research."""
    if date.month not in (3, 6, 9, 12) or date.weekday() != 4:
        return False
    return 15 <= date.day <= 21
