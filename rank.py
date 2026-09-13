"""Universe filtering, scoring and bucketing."""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd

import config as C


# ------------------------------------------------------------------ universe
# SIC code -> sector, at two granularities.
#
# `sic_to_major_group` (the 2-digit SIC major group, ~80 categories in
# practice) is the PRIMARY mapping, used as `industry` for clustering in
# evaluate(). Division-level (~10 categories, `sic_to_division`) is kept only
# as a secondary column: it is too coarse for clustering — "Manufacturing"
# alone spans food processing and defence electronics, so treating it as one
# cluster would flatten unrelated shocks into "one" shock.
#
# Major-group titles verified against the 1987 SIC manual structure; code ->
# major-group arithmetic verified against SEC's own SIC code list (e.g. SIC
# 3571 "Electronic Computers" falls under major group 35 "Machinery, Except
# Electrical", not 36 "Electrical & Electronic Equipment" — an artifact of the
# 1987 classification, not a bug here).
_SIC_MAJOR_GROUPS = {
    "01": "Agricultural Production - Crops",
    "02": "Agricultural Production - Livestock & Animal Specialties",
    "07": "Agricultural Services",
    "08": "Forestry",
    "09": "Fishing, Hunting & Trapping",
    "10": "Metal Mining",
    "12": "Bituminous Coal & Lignite Mining",
    "13": "Crude Petroleum & Natural Gas",
    "14": "Mining & Quarrying of Nonmetallic Minerals",
    "15": "General Building Contractors",
    "16": "Heavy Construction",
    "17": "Construction - Special Trade Contractors",
    "20": "Food & Kindred Products",
    "21": "Tobacco Products",
    "22": "Textile Mill Products",
    "23": "Apparel & Other Finished Products",
    "24": "Lumber & Wood Products",
    "25": "Furniture & Fixtures",
    "26": "Paper & Allied Products",
    "27": "Printing & Publishing",
    "28": "Chemicals & Allied Products",
    "29": "Petroleum Refining & Related Industries",
    "30": "Rubber & Miscellaneous Plastics Products",
    "31": "Leather & Leather Products",
    "32": "Stone, Clay, Glass & Concrete Products",
    "33": "Primary Metal Industries",
    "34": "Fabricated Metal Products",
    "35": "Machinery, Except Electrical",
    "36": "Electrical & Electronic Equipment",
    "37": "Transportation Equipment",
    "38": "Measuring, Analyzing & Controlling Instruments",
    "39": "Miscellaneous Manufacturing Industries",
    "40": "Railroad Transportation",
    "41": "Local & Suburban Transit",
    "42": "Trucking & Warehousing",
    "44": "Water Transportation",
    "45": "Air Transportation",
    "46": "Pipelines, Except Natural Gas",
    "47": "Transportation Services",
    "48": "Communications",
    "49": "Electric, Gas & Sanitary Services",
    "50": "Wholesale Trade - Durable Goods",
    "51": "Wholesale Trade - Nondurable Goods",
    "52": "Retail - Building Materials & Garden Supply",
    "53": "Retail - General Merchandise Stores",
    "54": "Retail - Food Stores",
    "55": "Retail - Auto Dealers & Gasoline Stations",
    "56": "Retail - Apparel & Accessory Stores",
    "57": "Retail - Home Furniture & Furnishings",
    "58": "Retail - Eating & Drinking Places",
    "59": "Retail - Miscellaneous Retail",
    "60": "Depository Institutions",
    "61": "Nondepository Credit Institutions",
    "62": "Security & Commodity Brokers",
    "63": "Insurance Carriers",
    "64": "Insurance Agents & Brokers",
    "65": "Real Estate",
    "67": "Holding & Other Investment Offices",
    "70": "Hotels & Other Lodging Places",
    "72": "Personal Services",
    "73": "Business Services",
    "75": "Automotive Repair & Services",
    "76": "Miscellaneous Repair Services",
    "78": "Motion Picture & Video Services",
    "79": "Amusement & Recreation Services",
    "80": "Health Services",
    "81": "Legal Services",
    "82": "Educational Services",
    "83": "Social Services",
    "86": "Membership Organizations",
    "87": "Engineering & Management Services",
    "88": "American Depositary Receipts & Foreign Governments",
    "89": "Services, NEC",
    "97": "Public Administration / International Affairs",
    "99": "Nonclassifiable Establishments",
}

_SIC_DIVISIONS = [
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


def sic_to_major_group(sic) -> str:
    """Map a SIC code to its 2-digit major group (~80 categories in practice).
    NaN/unmapped -> 'Unknown'. Zero-pads to 4 digits before slicing — SEC
    returns codes like 100 for what the manual prints as "0100"; slicing the
    unpadded string would give the wrong major group (e.g. "10" for
    Agriculture instead of "01")."""
    if sic is None or pd.isna(sic):
        return "Unknown"
    code = f"{int(sic):04d}"
    group = code[:2]
    title = _SIC_MAJOR_GROUPS.get(group)
    return f"{group} - {title}" if title else "Unknown"


def sic_to_division(sic) -> str:
    """Map a 4-digit SIC code to a coarse (~10-category) division. Kept as a
    secondary column only — too coarse to cluster on, see module notes above."""
    if sic is None or pd.isna(sic):
        return "Unknown"
    sic = int(sic)
    for lo, hi, name in _SIC_DIVISIONS:
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
            sic = out["ticker"].map(ref.set_index("ticker")["sic_code"])
            out["industry"] = sic.map(sic_to_major_group)
            out["sic_division"] = sic.map(sic_to_division)
        else:
            out["industry"] = "Unknown"
            out["sic_division"] = "Unknown"
    else:
        out["market_cap"] = np.nan
        out["industry"] = "Unknown"
        out["sic_division"] = "Unknown"

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
