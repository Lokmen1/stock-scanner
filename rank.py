"""Universe filtering, scoring and bucketing."""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd

import config as C


# ------------------------------------------------------------------ universe
# SIC code -> coarse sector, using the standard SIC division ranges. Coarse by
# design: this is for grouping signals in `evaluate`, not sector-fund-grade
# classification.
_SIC_SECTORS = [
    (100, 999, "Agriculture"),
    (1000, 1499, "Mining"),
    (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"),
    (4000, 4999, "Transportation/Utilities"),
    (5000, 5199, "Wholesale Trade"),
    (5200, 5999, "Retail Trade"),
    (6000, 6799, "Finance/Insurance/Real Estate"),
    (7000, 8999, "Services"),
    (9100, 9999, "Public Administration"),
]


def sic_to_sector(sic) -> str:
    """Map a 4-digit SIC code to a coarse sector. NaN/unmapped -> 'Unknown'."""
    if sic is None or pd.isna(sic):
        return "Unknown"
    sic = int(sic)
    for lo, hi, name in _SIC_SECTORS:
        if lo <= sic <= hi:
            return name
    return "Unknown"


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
        if "sic_code" in ref.columns:
            sic = ref.set_index("ticker")["sic_code"]
            out["industry"] = out["ticker"].map(sic).map(sic_to_sector)
        else:
            out["industry"] = "Unknown"
    else:
        out["market_cap"] = np.nan
        out["industry"] = "Unknown"

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
    # already visible on every free screener. Deliberately NOT called
    # "accumulation" — there is a seller for every buyer, and this pattern is
    # equally consistent with distribution, block crossing, or index flow.
    high_participation = d[(d["vol_z"] >= C.Z_VOLUME_STRONG) & (d["ret_z"].abs() <= C.Z_RETURN_QUIET)]

    return {
        "momentum": momentum.nlargest(C.MAX_PER_BUCKET, "score"),
        "dislocation": dislocation.nlargest(C.MAX_PER_BUCKET, "score"),
        "high_participation_low_resolution": high_participation.nlargest(C.MAX_PER_BUCKET, "vol_z"),
    }


def raw_crossings(d: pd.DataFrame) -> dict[str, int]:
    """Names crossing each gate before MAX_PER_BUCKET truncates the list.

    Mirrors the exact gate conditions in `bucket()` above — read-only, changes
    nothing about what qualifies. Lets the report show how loose or tight
    today actually was relative to the cap, and relative to its own history.
    """
    d = d[~d["suspect_corp_action"]]
    momentum = (d["ret_z"] >= C.Z_RETURN_GATE) & (d["vol_z"] >= C.Z_VOLUME_GATE)
    dislocation = (d["ret_z"] <= -C.Z_RETURN_GATE) & (d["vol_z"] >= C.Z_VOLUME_GATE)
    high_participation = (d["vol_z"] >= C.Z_VOLUME_STRONG) & (d["ret_z"].abs() <= C.Z_RETURN_QUIET)
    return {
        "momentum": int(momentum.sum()),
        "dislocation": int(dislocation.sum()),
        "high_participation_low_resolution": int(high_participation.sum()),
    }
