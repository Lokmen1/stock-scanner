"""Feature computation.

Design notes on the two choices that matter most:

ROBUST SCALE. The naive z-score divides by a 60-day sample standard deviation.
A stock that gaps 9% every quarter carries an inflated sigma for the following
three months, so its next genuine surprise is scored DOWN precisely because it
has a history of surprises. Using median and MAD instead makes the location and
scale estimates insensitive to the handful of jump days inside the window, which
is exactly the behaviour you want for an earnings-heavy universe.

MARKET PROXY. Rather than requiring an SPY feed, the market return for a session
is the cross-sectional median return of the investable universe. It is available
by construction from the same grouped-daily file, it is robust to a few names
blowing up, and for a relative-strength feature it is close enough to a
cap-weighted index that the difference does not change the ranking materially.
Sector-relative is the next upgrade and needs a sector map (SIC -> sector).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _robust_z(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    """(x - rolling median) / (rolling MAD * 1.4826), shifted to exclude today."""
    hist = s.shift(1)                                   # today must not inform its own baseline
    med = hist.rolling(window, min_periods=min_obs).median()
    mad = (hist - med).abs().rolling(window, min_periods=min_obs).median()
    scale = mad * C.MAD_SCALE
    scale = scale.where(scale > 1e-9)                   # avoid divide-by-zero on flat series
    return (s - med) / scale


def _classic_z(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    hist = s.shift(1)
    mu = hist.rolling(window, min_periods=min_obs).mean()
    sd = hist.rolling(window, min_periods=min_obs).std()
    sd = sd.where(sd > 1e-9)
    return (s - mu) / sd


def compute(df: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add all features. Expects columns: date, ticker, open, high, low, close, volume.

    `ref` (universe reference data, with a shares_outstanding column) is
    optional and only needed for float_turnover — pass the same `ref` used
    for apply_universe(), or omit it and that one column comes back NaN.
    """
    z = _robust_z if C.ROBUST_SCALE else _classic_z
    df = df.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", sort=False)

    # ---- returns -----------------------------------------------------------
    df["prev_close"] = g["close"].shift(1)
    df["ret_1d"] = df["close"] / df["prev_close"] - 1.0

    # ---- dollar volume -----------------------------------------------------
    df["dollar_volume"] = df["close"] * df["volume"]
    df["log_dv"] = np.log(df["dollar_volume"].clip(lower=1.0))
    df["adv_20"] = g["dollar_volume"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).mean()
    )

    # ---- z-scores ----------------------------------------------------------
    df["ret_z"] = g["ret_1d"].transform(lambda s: z(s, C.LOOKBACK, C.MIN_OBS))
    df["vol_z"] = g["log_dv"].transform(lambda s: z(s, C.LOOKBACK, C.MIN_OBS))

    # ---- gap ---------------------------------------------------------------
    df["gap"] = df["open"] / df["prev_close"] - 1.0
    df["gap_z"] = g["gap"].transform(lambda s: z(s, C.LOOKBACK, C.MIN_OBS))
    # Did the session extend the gap or fade it? Different information.
    df["intraday"] = df["close"] / df["open"] - 1.0
    df["gap_follow"] = np.sign(df["gap"]) * np.sign(df["intraday"])

    # ---- breakouts ---------------------------------------------------------
    for w in C.BREAKOUT_WINDOWS:
        hi = g["close"].transform(lambda x: x.shift(1).rolling(w, min_periods=w // 2).max())
        lo = g["close"].transform(lambda x: x.shift(1).rolling(w, min_periods=w // 2).min())
        df[f"bo_{w}"] = df["close"] / hi - 1.0
        df[f"bd_{w}"] = df["close"] / lo - 1.0
    df["breakout"] = df[[f"bo_{w}" for w in C.BREAKOUT_WINDOWS]].max(axis=1)
    df["breakout_label"] = _breakout_label(df)
    # Distance from the highest close anywhere in the STORED history — not
    # literally all-time, just as far back as this run's price data reaches.
    ath = g["close"].transform(lambda x: x.shift(1).expanding().max())
    df["bo_all"] = df["close"] / ath - 1.0

    # ---- market-relative return -------------------------------------------
    mkt = df.groupby("date")["ret_1d"].transform("median")
    df["mkt_ret"] = mkt
    df["rel_ret"] = df["ret_1d"] - mkt

    # ---- corporate-action guard -------------------------------------------
    # A huge move on ordinary volume is almost always an unhandled corporate
    # action, not information. Flag rather than drop, so you can inspect.
    df["suspect_corp_action"] = (
        (df["ret_1d"].abs() > C.CORP_ACTION_RET_GUARD)
        & (df["vol_z"].fillna(0) < C.CORP_ACTION_VOL_FLOOR)
    )

    # ---- continuation features ----------------------------------------------
    # Deterministic columns only. NOT inputs to score_day's weighted blend —
    # no continuation score, no expected move, no price target. The point is
    # to log enough shape/context that weights can be FITTED from forward
    # returns later (evaluate()'s job), not asserted now.

    # Close Location Value: where in the day's range the close landed.
    # 1.0 = closed at the high, 0.0 = closed at the low. Undefined (NaN) on a
    # zero-range session.
    day_range = df["high"] - df["low"]
    df["clv"] = ((df["close"] - df["low"]) / day_range).where(day_range > 1e-9)

    # Gap retention: of the overnight gap (open vs prior close), how much
    # survived to the close? >1 means the session extended the gap, <0 means
    # it fully reversed it. Undefined on a negligible gap — a few-cent open
    # vs prior close would make this ratio noise, not signal.
    gap_abs = df["open"] - df["prev_close"]
    min_gap = C.GAP_RETENTION_MIN_GAP * df["prev_close"].abs()
    df["gap_retention"] = ((df["close"] - df["prev_close"]) / gap_abs).where(gap_abs.abs() > min_gap)

    # RVOL: recent dollar volume (1/3/5-day, including today) against the
    # prior-20-day baseline already computed as adv_20.
    df["rvol_1"] = df["dollar_volume"] / df["adv_20"]
    df["rvol_3"] = g["dollar_volume"].transform(lambda x: x.rolling(3).mean()) / df["adv_20"]
    df["rvol_5"] = g["dollar_volume"].transform(lambda x: x.rolling(5).mean()) / df["adv_20"]

    # Pre-event compression: was the name unusually quiet right before today,
    # relative to its own longer-run range? Both windows are shifted to
    # exclude today — this describes the setup BEFORE the trigger, not today
    # itself.
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["prev_close"]).abs(),
        (df["low"] - df["prev_close"]).abs(),
    ], axis=1).max(axis=1)
    df["true_range"] = true_range
    atr_short = g["true_range"].transform(
        lambda x: x.shift(1).rolling(C.ATR_SHORT_WINDOW).mean())
    atr_long = g["true_range"].transform(
        lambda x: x.shift(1).rolling(C.ATR_LONG_WINDOW).mean())
    df["pre_event_compression"] = atr_short / atr_long.where(atr_long > 1e-9)

    # Float turnover: only where the universe reference data supplies a share
    # count. Missing share counts -> NaN, not a silent zero.
    if ref is not None and "shares_outstanding" in ref.columns:
        shares = df["ticker"].map(ref.set_index("ticker")["shares_outstanding"])
        df["float_turnover"] = (df["volume"] / shares).where(shares > 0)
    else:
        df["float_turnover"] = np.nan

    return df


def _breakout_label(df: pd.DataFrame) -> pd.Series:
    label = pd.Series("", index=df.index, dtype=object)
    for w, name in zip(C.BREAKOUT_WINDOWS, ("20D", "60D", "52W")):
        label = label.mask((df[f"bo_{w}"] > 0) & (label == ""), name)
    # longest window wins
    for w, name in zip(reversed(C.BREAKOUT_WINDOWS), ("52W", "60D", "20D")):
        label = label.mask(df[f"bo_{w}"] > 0, name)
    return label.replace("", "—")


def feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: how much do the five inputs overlap?

    Gap and return are mechanically correlated on gap days — a stock that opens
    +6% and closes +7% scores twice for one event. If the off-diagonal terms are
    above ~0.7 you are weighting one signal three times, not blending three.
    """
    cols = ["ret_z", "vol_z", "rel_ret", "gap_z", "breakout"]
    return df[cols].corr(method="spearman")
