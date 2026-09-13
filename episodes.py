"""Event-episode tracking.

The unit of analysis is an EPISODE — a persisting situation in one name — not
a ticker-day. A ticker's trigger extends its open episode if the previous
trigger was within GAP_SESSIONS trading sessions; a longer gap closes that
episode and starts a new one.

The episode id is derived from the ticker and its first trigger date
(f"{ticker}-{first_trigger_date:%Y%m%d}"), not from a counter. That means
episodes never need separate persisted state: replaying the signal log from
scratch always reproduces the same ids.

Replaces the old `streak` field (rank.py's add_persistence), which counted
appearances in a fixed trailing calendar window without session-aware gaps
and without distinguishing one episode from a later, unrelated one for the
same ticker. `days_triggered` is its direct successor.
"""
from __future__ import annotations

import pandas as pd

GAP_SESSIONS = 5


def _session_gap(from_date: pd.Timestamp, to_date: pd.Timestamp,
                  trading_days: pd.DatetimeIndex) -> int:
    """Number of trading sessions between two dates in the real calendar."""
    i0 = trading_days.searchsorted(from_date)
    i1 = trading_days.searchsorted(to_date)
    return int(i1 - i0)


def assign_episodes(today: pd.DataFrame, history: pd.DataFrame,
                     trading_days: pd.DatetimeIndex, date: pd.Timestamp) -> pd.DataFrame:
    """Add event_id/event_age/days_triggered/max_score/first_trigger_date to
    today's flagged rows.

    `today` needs columns ticker, score. `history` is the signal log written
    so far (needs ticker, date, event_id, first_trigger_date, days_triggered,
    max_score) — the same columns this function produces, so the log is
    self-describing across days.
    """
    if today.empty:
        out = today.copy()
        for col in ("event_id", "event_age", "days_triggered", "max_score", "first_trigger_date"):
            out[col] = pd.Series(dtype=object)
        return out

    # Signal-log rows written before this feature existed won't have these
    # columns. Treat them as having no usable episode data rather than crash
    # or fabricate continuity — every ticker just opens a fresh episode as of
    # today, which is the only honest thing to do with a pre-episode history.
    required = {"event_id", "first_trigger_date", "days_triggered", "max_score"}
    has_episode_cols = not history.empty and required.issubset(history.columns)

    out = today.copy()
    event_ids, event_ages, days_triggered, max_scores, first_dates = [], [], [], [], []

    for _, row in out.iterrows():
        t = row["ticker"]
        prior = history[history["ticker"] == t] if has_episode_cols else history.iloc[0:0]

        if prior is None or prior.empty:
            gap = None
        else:
            last = prior.sort_values("date").iloc[-1]
            gap = _session_gap(last["date"], date, trading_days)

        if gap is None or gap > GAP_SESSIONS:
            first_trigger_date = date
            eid = f"{t}-{date:%Y%m%d}"
            n_triggered = 1
            m_score = row["score"]
            age = 0
        else:
            first_trigger_date = last["first_trigger_date"]
            eid = last["event_id"]
            n_triggered = int(last["days_triggered"]) + 1
            m_score = max(float(last["max_score"]), row["score"])
            age = _session_gap(first_trigger_date, date, trading_days)

        event_ids.append(eid)
        event_ages.append(age)
        days_triggered.append(n_triggered)
        max_scores.append(m_score)
        first_dates.append(first_trigger_date)

    out["event_id"] = event_ids
    out["event_age"] = event_ages
    out["days_triggered"] = days_triggered
    out["max_score"] = max_scores
    out["first_trigger_date"] = first_dates
    return out
