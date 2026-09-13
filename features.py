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


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add all features. Expects columns: date, ticker, open, high, low, close, volume."""
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
