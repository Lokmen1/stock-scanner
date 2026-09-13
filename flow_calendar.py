"""Explicit calendar of days when known, mechanical (non-fundamental) order
flow is expected — index reconstitutions/rebalances and options expiries.

Forced flow (an index fund must buy or sell because a name entered or left a
benchmark, not because anyone re-priced anything) is real information about
supply and demand. It is not fundamental information, and conflating the two
by EXCLUDING these days would throw away real data. So this module tags, it
does not filter: `mechanical_flow_possible(date)` feeds a boolean column on
the signal log. Nothing is excluded from scoring, gating or bucketing.

Sources, verified September 2026 (each an explicit, named rule — not a single
vague day-range heuristic):
  - S&P Dow Jones quarterly rebalance: effective 3rd Friday of Mar/Jun/Sep/Dec
    (same day as equity-index triple/quadruple witching, by design).
  - FTSE Russell US Indexes reconstitution: annual in June through 2025;
    SEMI-ANNUAL from 2026 (LSEG/FTSE Russell, "Russell US Indexes to Move to
    a Semi-Annual Reconstitution Frequency from 2026").
      * June leg: effective after the close, LAST Friday of June — this is
        the date behind the 223-name momentum-crossing outlier on 2026-06-26.
      * December leg (2026 onward): effective after the close, 2nd Friday of
        December (moved from an originally-planned 2nd Friday of November;
        FTSE Russell schedule update, 5 Nov 2025).
  - Nasdaq-100 annual reconstitution: effective at the open on the Monday
    following the 3rd Friday of December.
  - MSCI Quarterly / Semi-Annual Index Review: changes as of the close of the
    LAST trading day of Feb/May/Aug/Nov (Feb/Aug = quarterly; May/Nov =
    semi-annual, the larger of the two).
  - Standard equity options expiry: 3rd Friday of every month. Mar/Jun/Sep/Dec
    additionally carry index option/future expiry ("triple/quadruple
    witching"), already covered by the S&P line above.

Crude on holidays, same as the rest of this project (see README's known
limitations): a computed date that happens to be a market holiday is not
special-cased. Index providers usually shift such dates by a day; this module
does not attempt to model that.
"""
from __future__ import annotations

import calendar
import datetime as dt

import pandas as pd

FRIDAY = 4


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th (1-indexed) occurrence of `weekday` (Mon=0..Sun=6) in month."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    """The LAST occurrence of `weekday` in year/month."""
    last_day = calendar.monthrange(year, month)[1]
    last = dt.date(year, month, last_day)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _last_business_day(year: int, month: int) -> dt.date:
    """Last Mon-Fri date in the month (does not know about market holidays)."""
    last_day = calendar.monthrange(year, month)[1]
    d = dt.date(year, month, last_day)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def known_flow_events(date: pd.Timestamp | dt.date) -> list[str]:
    """Names of every known mechanical-flow event active on `date`. Empty list
    if none. Callers should only pass real trading sessions."""
    d = date.date() if hasattr(date, "date") else date
    y, mo = d.year, d.month
    events: list[str] = []

    third_friday = _nth_weekday(y, mo, FRIDAY, 3)
    if d == third_friday:
        events.append("monthly equity options expiry")
        if mo in (3, 6, 9, 12):
            events.append("S&P quarterly rebalance / triple witching")

    if mo == 12 and d == third_friday + dt.timedelta(days=3):
        events.append("Nasdaq-100 annual reconstitution (effective)")

    if mo == 6 and d == _last_weekday(y, mo, FRIDAY):
        events.append("FTSE Russell US Indexes reconstitution (June)")
    if y >= 2026 and mo == 12 and d == _nth_weekday(y, mo, FRIDAY, 2):
        events.append("FTSE Russell US Indexes reconstitution (December, semi-annual)")

    if mo in (2, 5, 8, 11) and d == _last_business_day(y, mo):
        label = "semi-annual" if mo in (5, 11) else "quarterly"
        events.append(f"MSCI {label} index review (effective)")

    return events


def mechanical_flow_possible(date: pd.Timestamp | dt.date) -> bool:
    return bool(known_flow_events(date))
